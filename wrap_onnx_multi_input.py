#!/usr/bin/env python3
"""
ONNX 模型包装脚本：将单输入 ONNX (input [B, 738] → output [B, 29])
包装为多输入 ONNX，每个 observation term 作为独立输入。

★ 关键: use_gym_history: true 时，C++ ObservationManager 的输出是
   逐时间步交织 (per-time-step interleaving)，而非逐观测项分组。

   原始模型输入顺序:
   [t-5: all_terms(123)] [t-4: all_terms(123)] ... [t: all_terms(123)] = 738

   包装逻辑:
   7 个 per-term 输入 → Reshape 分离时间步 → Concat 交织 → Reshape 展平 → 原始模型

使用方式:
    python wrap_onnx_multi_input.py

输入:  policy_name.onnx (单输入, input [B, 738] → output [B, 29])
输出: policy_name_multi.onnx (多输入, 7 个独立 observation term)
"""

import onnx
from onnx import helper, TensorProto
import numpy as np

# =============================================================================
# 1. 定义 Observation Term 拆分方案 (与 physhsi_deploy.yaml 一致)
# =============================================================================
# use_gym_history: true, history_length: 6
# C++ ObservationManager 为每个 group 产生 per-term 时间堆叠输出:
#   例如 base_ang_vel: [ang_t5(3)|ang_t4(3)|ang_t3(3)|ang_t2(3)|ang_t1(3)|ang_t0(3)] = 18 dims
#
# 原始模型期望 per-time-step 交织:
#   [ang_t5|grav_t5|jpos_t5|jvel_t5|ee_t5|act_t5|task_t5|  (t-5: 123)
#    ang_t4|grav_t4|jpos_t4|jvel_t4|ee_t4|act_t4|task_t4|  (t-4: 123)
#    ...                                                      ...
#    ang_t0|grav_t0|jpos_t0|jvel_t0|ee_t0|act_t0|task_t0]  (t: 123)
#
# 顺序必须与用户指定的 738 维组成严格一致:
#   偏移 0-122:   t-5 (最新 = h=0 in C++ history)
#   偏移 123-245: t-4
#   偏移 246-368: t-3
#   偏移 369-491: t-2
#   偏移 492-614: t-1
#   偏移 615-737: t (最旧 = h=5 in C++ history)

HISTORY_LENGTH = 6
OBS_TERMS = [
    # (ONNX 输入名称, 单帧维度)
    # ★ 顺序必须与用户在每步 123 维中的排列一致 ★
    ("base_ang_vel",       3),   # 偏移 0-2
    ("projected_gravity",  3),   # 偏移 3-5
    ("joint_pos_rel",     29),   # 偏移 6-34
    ("joint_vel_rel",     29),   # 偏移 35-63
    ("end_effector_pos",  15),   # 偏移 64-78
    ("last_action",       29),   # 偏移 79-107
    ("task_obs",          15),   # 偏移 108-122
]
PER_STEP_DIM = sum(dim for _, dim in OBS_TERMS)  # 123
TOTAL_DIM = PER_STEP_DIM * HISTORY_LENGTH          # 738

assert PER_STEP_DIM == 123, f"每步维度应为 123，实际为 {PER_STEP_DIM}"
assert TOTAL_DIM == 738, f"总维度应为 738，实际为 {TOTAL_DIM}"
print(f"每步维度: {PER_STEP_DIM} ✓")
print(f"总观测维度: {TOTAL_DIM} ✓")

# =============================================================================
# 2. 加载原始 ONNX 模型
# =============================================================================
INPUT_MODEL = "policy_name.onnx"
OUTPUT_MODEL = "policy_name_multi.onnx"

original_model = onnx.load(INPUT_MODEL)
print(f"\n加载原始模型: {INPUT_MODEL}")
print(f"  输入: {[inp.name for inp in original_model.graph.input]}")
print(f"  输出: {[out.name for out in original_model.graph.output]}")

original_graph = original_model.graph
orig_input = original_graph.input[0]   # 'input': [B, 738]
orig_output = original_graph.output[0] # 'output': [B, 29]

# =============================================================================
# 3. 创建多输入包装图 — 核心: Reshape → Concat → Reshape 实现交织
# =============================================================================

# --- 3.1 创建 7 个新输入 ---
new_inputs = []
reshape_outputs = []  # 每个输入 Reshape 后的中间结果名称

for term_name, dim in OBS_TERMS:
    input_dim = dim * HISTORY_LENGTH
    new_input = helper.make_tensor_value_info(
        term_name,
        TensorProto.FLOAT,
        ["batch_size", input_dim]
    )
    new_inputs.append(new_input)

# --- 3.2 为每个输入创建 Reshape [B, dim×6] → [B, 6, dim] ---
# 使用 initializer 提供 reshape 的目标形状
new_nodes = []
new_initializers = list(original_graph.initializer)  # 保留原始权重

for i, (term_name, dim) in enumerate(OBS_TERMS):
    # Reshape 目标形状: [batch_size, 6, dim]
    # ONNX Reshape 支持用 0 表示"保持原维度", -1 表示"推断"
    # 这里我们用显式值: shape = [0, 6, dim] 或更安全地使用常量
    shape_name = f"reshape_shape_{i}"

    # 创建形状常量 initializer [3]: [batch_size, 6, dim]
    # 用 0 表示 batch_size 维度保持动态
    shape_data = np.array([0, HISTORY_LENGTH, dim], dtype=np.int64)
    shape_init = helper.make_tensor(
        name=shape_name,
        data_type=TensorProto.INT64,
        dims=[3],
        vals=shape_data.tobytes(),
        raw=True
    )
    new_initializers.append(shape_init)

    reshape_output = f"reshape_{term_name}"
    reshape_node = helper.make_node(
        'Reshape',
        inputs=[term_name, shape_name],
        outputs=[reshape_output],
        name=f'reshape_{term_name}'
    )
    new_nodes.append(reshape_node)
    reshape_outputs.append(reshape_output)

# --- 3.3 Concat 所有 Reshape 输出在 axis=2 上 → [B, 6, 123] ---
concat_node = helper.make_node(
    'Concat',
    inputs=reshape_outputs,
    outputs=['interleaved_obs'],
    axis=2,  # 在最后一维拼接: 3+3+29+29+15+29+15 = 123
    name='interleave_concat'
)
new_nodes.append(concat_node)

# --- 3.4 Reshape [B, 6, 123] → [B, 738] ---
# 行主序展平: row0(t-5) | row1(t-4) | ... | row5(t)
final_shape_name = "final_reshape_shape"
final_shape_data = np.array([0, TOTAL_DIM], dtype=np.int64)  # [batch_size, 738]
final_shape_init = helper.make_tensor(
    name=final_shape_name,
    data_type=TensorProto.INT64,
    dims=[2],
    vals=final_shape_data.tobytes(),
    raw=True
)
new_initializers.append(final_shape_init)

final_reshape = helper.make_node(
    'Reshape',
    inputs=['interleaved_obs', final_shape_name],
    outputs=['obs_flat_738'],
    name='final_reshape'
)
new_nodes.append(final_reshape)

# =============================================================================
# 4. 将原始模型的节点追加进来，输入名替换为包装后的输出
# =============================================================================
for node in original_graph.node:
    new_node = onnx.NodeProto()
    new_node.CopyFrom(node)
    # 将引用 'input' 的地方替换为 'obs_flat_738'
    new_input_names = []
    for inp_name in node.input:
        if inp_name == orig_input.name:
            new_input_names.append('obs_flat_738')
        else:
            new_input_names.append(inp_name)
    del new_node.input[:]
    new_node.input.extend(new_input_names)
    new_nodes.append(new_node)

# =============================================================================
# 5. 创建输出
# =============================================================================
new_output = helper.make_tensor_value_info(
    orig_output.name,
    TensorProto.FLOAT,
    ["batch_size", 29]
)

# =============================================================================
# 6. 构建新的 Graph 和 Model
# =============================================================================
new_graph = helper.make_graph(
    nodes=new_nodes,
    name="multi_input_interleaved",
    inputs=new_inputs,
    outputs=[new_output],
    initializer=new_initializers,
)

new_model = helper.make_model(
    new_graph,
    producer_name="unitree_rl_lab_interleaved_wrapper",
    opset_imports=[onnx.helper.make_opsetid("", 11)],
    ir_version=6,
)

# =============================================================================
# 7. 验证并保存
# =============================================================================
onnx.checker.check_model(new_model)
print("\nONNX 模型结构验证通过 ✓")
onnx.save(new_model, OUTPUT_MODEL)
print(f"已保存多输入模型: {OUTPUT_MODEL}")

# =============================================================================
# 8. 打印最终结构
# =============================================================================
print("\n" + "=" * 60)
print("包装后 ONNX 模型结构")
print("=" * 60)
print(f"\n输入 ({len(new_model.graph.input)} 个):")
for inp in new_model.graph.input:
    shape = [d.dim_param if d.dim_param else d.dim_value
             for d in inp.type.tensor_type.shape.dim]
    print(f"  '{inp.name}': shape={shape}")

print(f"\n输出 ({len(new_model.graph.output)} 个):")
for out in new_model.graph.output:
    shape = [d.dim_param if d.dim_param else d.dim_value
             for d in out.type.tensor_type.shape.dim]
    print(f"  '{out.name}': shape={shape}")

print(f"\n包装节点: {len(new_nodes)} (7 Reshape + 1 Concat + 1 Reshape + 7 原始)")
print(f"初始化器/权重: {len(new_model.graph.initializer)} 个")

# =============================================================================
# 9. 运行时验证 — 正确交织 vs 原始单输入
# =============================================================================
print("\n" + "=" * 60)
print("运行时推理验证 (正确交织 vs 原始模型)")
print("=" * 60)

try:
    import onnxruntime as ort

    wrapped_session = ort.InferenceSession(OUTPUT_MODEL)
    orig_session = ort.InferenceSession(INPUT_MODEL)

    # --- 9.1 生成 per-term 时间堆叠的测试输入 ---
    # C++ ObservationManager 为每个 group 产生:
    #   [t-5_data | t-4_data | t-3_data | t-2_data | t-1_data | t_data]
    # 其中 h=0 是最老帧 (t-5), h=5 是最新帧 (t)
    per_term_inputs = {}
    raw_frames = []  # 按时间步存储各 term 的原始值

    np.random.seed(42)
    for step in range(HISTORY_LENGTH):
        frame = {}
        for term_name, dim in OBS_TERMS:
            frame[term_name] = np.random.randn(dim).astype(np.float32)
        raw_frames.append(frame)

    # 构造 C++ 侧 per-term 时间堆叠输入
    # 例如 base_ang_vel: [t-5(3) | t-4(3) | t-3(3) | t-2(3) | t-1(3) | t(3)] = 18
    for term_name, dim in OBS_TERMS:
        stacked = []
        for step in range(HISTORY_LENGTH):  # h=0 (t-5) to h=5 (t)
            stacked.append(raw_frames[step][term_name])
        per_term_inputs[term_name] = np.concatenate(stacked).reshape(1, -1).astype(np.float32)

    # --- 9.2 运行包装后的模型 ---
    wrapped_output = wrapped_session.run(None, per_term_inputs)

    # --- 9.3 构造正确的 per-time-step 交织输入与原始模型对比 ---
    # 正确顺序: [t-5_all123 | t-4_all123 | t-3_all123 | t-2_all123 | t-1_all123 | t_all123]
    correct_flat = []
    for step in range(HISTORY_LENGTH):  # t-5 first, t last
        for term_name, dim in OBS_TERMS:
            correct_flat.append(raw_frames[step][term_name])
    correct_input = np.concatenate(correct_flat).reshape(1, -1).astype(np.float32)

    orig_output = orig_session.run(None, {"input": correct_input})

    # --- 9.4 对比 ---
    diff = np.abs(wrapped_output[0] - orig_output[0])
    print(f"\n  包装模型输出形状: {wrapped_output[0].shape}")
    print(f"  原始模型输出形状: {orig_output[0].shape}")
    print(f"  最大误差: {diff.max():.6e}")
    print(f"  均方误差: {np.mean(diff**2):.6e}")

    if diff.max() < 1e-5:
        print("  ✓ 输出完全一致 — 交织逻辑正确")
    else:
        print(f"  ✗ 存在差异!")

    # --- 9.5 额外验证: 错误排序对比 (确认旧版 bug) ---
    print("\n" + "-" * 40)
    print("对比: 如果用逐项分组 (旧版错误做法) 会怎样?")
    wrong_flat = []
    for term_name, dim in OBS_TERMS:
        for step in range(HISTORY_LENGTH):
            wrong_flat.append(raw_frames[step][term_name])
    wrong_input = np.concatenate(wrong_flat).reshape(1, -1).astype(np.float32)
    wrong_output = orig_session.run(None, {"input": wrong_input})
    wrong_diff = np.abs(wrong_output[0] - orig_output[0])
    print(f"  错误排序 → 原始模型输出的最大误差: {wrong_diff.max():.6e}")
    print(f"  错误排序 → 原始模型输出的均方误差: {np.mean(wrong_diff**2):.6e}")
    if wrong_diff.max() > 0.01:
        print("  ✓ 确认旧版 wrapper 会产生完全错误的输出!")

except ImportError:
    print("\n  (跳过运行时验证: onnxruntime 未安装)")
except Exception as e:
    import traceback
    print(f"\n  ⚠ 运行时验证失败: {e}")
    traceback.print_exc()
