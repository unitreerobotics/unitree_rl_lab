// =============================================================================
// unitree_mujoco 侧 — 特权信息话题发布器 (mqueue Publisher)
//
// 话题名称: /physhsi_privileged_state (POSIX 消息队列)
// 消息体:   PrivilegedStateMsg (128 bytes, 固定大小)
//
// 集成方式: 在 unitree_mujoco/simulate/ 仿真主循环中, 每次 mj_step() 之后
//           调用 publish_privileged_state()。
//
// 头文件: 可直接复制本文件中的 struct 定义到 unitree_mujoco 项目,
//         或 include deploy/robots/g1_29dof/include/PrivilegedState.h
// =============================================================================

#include <mqueue.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstring>
#include <cstdio>
#include <errno.h>
#include <unistd.h>

// MJCF headers (根据 unitree_mujoco 实际路径调整)
#include <mujoco/mujoco.h>


// =============================================================================
// PrivilegedStateMsg — 与 g1_ctrl 侧定义严格一致 (128 bytes)
// =============================================================================

struct PrivilegedStateMsg
{
    float end_effector_pos_w[15];  // 5 bodies × 3 (60 bytes)
    float root_pos_w[3];           // root link world pos (12 bytes)
    float box_pos_w[3];            // box world pos (12 bytes)
    float box_quat_w[4];           // box quat wxyz (16 bytes)
    float box_size[3];             // box half-size [w,d,h] (12 bytes)
    float goal_pos_w[3];           // goal world pos (12 bytes)
    uint32_t sequence;             // frame counter (4 bytes)
    uint32_t padding;              // alignment (4 bytes)
};

static_assert(sizeof(PrivilegedStateMsg) == 128, "Msg size must be 128 bytes");


// =============================================================================
// 全局状态
// =============================================================================

static mqd_t g_mqd = -1;
static uint32_t g_sequence = 0;

// Body name IDs (缓存在 init 中)
static int g_body_left_palm   = -1;
static int g_body_right_palm  = -1;
static int g_body_left_ankle  = -1;
static int g_body_right_ankle = -1;
static int g_body_head        = -1;
static int g_body_root        = -1;
static int g_body_box         = -1;
static int g_site_goal        = -1;
static int g_geom_box         = -1;

static constexpr const char* TOPIC_NAME = "/physhsi_privileged_state";


// =============================================================================
// init_privileged_publisher — 创建话题 (仿真启动时调用一次)
// =============================================================================

bool init_privileged_publisher(mjModel* m)
{
    // ---- 缓存 MuJoCo body/site/geom IDs (启动时获取, 避免每帧查找) ----
    g_body_left_palm   = mj_name2id(m, mjOBJ_BODY, "left_palm_link");
    g_body_right_palm  = mj_name2id(m, mjOBJ_BODY, "right_palm_link");
    g_body_left_ankle  = mj_name2id(m, mjOBJ_BODY, "left_ankle_pitch_link");
    g_body_right_ankle = mj_name2id(m, mjOBJ_BODY, "right_ankle_pitch_link");
    g_body_head        = mj_name2id(m, mjOBJ_BODY, "d455_link");
    g_body_root        = mj_name2id(m, mjOBJ_BODY, "torso_link");
    g_body_box         = mj_name2id(m, mjOBJ_BODY, "box");
    g_site_goal        = mj_name2id(m, mjOBJ_SITE, "goal");

    // 箱子 geom (用于读取尺寸)
    if (g_body_box >= 0)
    {
        int geom_start = m->body_geomadr[g_body_box];
        int geom_num   = m->body_geomnum[g_body_box];
        if (geom_num > 0 && m->geom_type[geom_start] == mjGEOM_BOX)
        {
            g_geom_box = geom_start;
        }
    }

    // 验证关键 body
    const char* required[] = {"torso_link", "left_palm_link", "right_palm_link",
                              "left_ankle_pitch_link", "right_ankle_pitch_link", "d455_link"};
    int* ids[] = {&g_body_root, &g_body_left_palm, &g_body_right_palm,
                  &g_body_left_ankle, &g_body_right_ankle, &g_body_head};
    for (int i = 0; i < 6; ++i)
    {
        if (*ids[i] < 0)
        {
            printf("[PrivilegedPublisher] WARNING: body '%s' not found in MJCF\n", required[i]);
        }
    }

    // ---- 创建 POSIX 消息队列 (话题) ----
    struct mq_attr attr;
    attr.mq_flags   = 0;                          // 阻塞模式
    attr.mq_maxmsg  = 10;                         // 最多缓存 10 条 (约 200ms @ 50Hz)
    attr.mq_msgsize = sizeof(PrivilegedStateMsg); // 128 bytes
    attr.mq_curmsgs = 0;

    // 先删除旧话题 (mq_unlink), 再创建新话题
    mq_unlink(TOPIC_NAME);
    g_mqd = mq_open(TOPIC_NAME, O_CREAT | O_WRONLY, 0666, &attr);

    if (g_mqd < 0)
    {
        printf("[PrivilegedPublisher] mq_open failed (errno=%d: %s)\n",
               errno, strerror(errno));
        return false;
    }

    printf("[PrivilegedPublisher] topic '%s' created (max_msgs=%ld, msgsize=%ld)\n",
           TOPIC_NAME, attr.mq_maxmsg, attr.mq_msgsize);
    return true;
}


// =============================================================================
// publish_privileged_state — 发布一帧特权信息 (每个仿真步调用)
// =============================================================================

void publish_privileged_state(mjModel* m, mjData* d)
{
    if (g_mqd < 0) return;

    PrivilegedStateMsg msg;
    std::memset(&msg, 0, sizeof(msg));
    msg.sequence = g_sequence++;

    // ---- (A) 末端执行器世界坐标 (5 bodies × 3) ----
    auto copy_body_pos = [&](int body_id, float* dst)
    {
        if (body_id >= 0)
        {
            dst[0] = d->xpos[body_id * 3 + 0];
            dst[1] = d->xpos[body_id * 3 + 1];
            dst[2] = d->xpos[body_id * 3 + 2];
        }
    };
    copy_body_pos(g_body_left_palm,   msg.end_effector_pos_w + 0);
    copy_body_pos(g_body_right_palm,  msg.end_effector_pos_w + 3);
    copy_body_pos(g_body_left_ankle,  msg.end_effector_pos_w + 6);
    copy_body_pos(g_body_right_ankle, msg.end_effector_pos_w + 9);
    copy_body_pos(g_body_head,        msg.end_effector_pos_w + 12);

    // ---- (B) 根Link世界坐标 ----
    copy_body_pos(g_body_root, msg.root_pos_w);

    // ---- (C) 箱子世界坐标和姿态 ----
    if (g_body_box >= 0)
    {
        msg.box_pos_w[0] = d->xpos[g_body_box * 3 + 0];
        msg.box_pos_w[1] = d->xpos[g_body_box * 3 + 1];
        msg.box_pos_w[2] = d->xpos[g_body_box * 3 + 2];

        // MuJoCo xquat: [w, x, y, z] (与训练代码四元数顺序一致)
        msg.box_quat_w[0] = d->xquat[g_body_box * 4 + 0]; // w
        msg.box_quat_w[1] = d->xquat[g_body_box * 4 + 1]; // x
        msg.box_quat_w[2] = d->xquat[g_body_box * 4 + 2]; // y
        msg.box_quat_w[3] = d->xquat[g_body_box * 4 + 3]; // z
    }

    // ---- (D) 箱子尺寸 (静态, 从 mjModel 读取) ----
    if (g_geom_box >= 0)
    {
        // MuJoCo geom_size 是半边长 [half_width, half_depth, half_height]
        msg.box_size[0] = m->geom_size[g_geom_box * 3 + 0];
        msg.box_size[1] = m->geom_size[g_geom_box * 3 + 1];
        msg.box_size[2] = m->geom_size[g_geom_box * 3 + 2];
    }

    // ---- (E) 目标点世界坐标 ----
    if (g_site_goal >= 0)
    {
        msg.goal_pos_w[0] = d->site_xpos[g_site_goal * 3 + 0];
        msg.goal_pos_w[1] = d->site_xpos[g_site_goal * 3 + 1];
        msg.goal_pos_w[2] = d->site_xpos[g_site_goal * 3 + 2];
    }

    // ---- (F) 发布消息到话题 ----
    if (mq_send(g_mqd, reinterpret_cast<const char*>(&msg),
                sizeof(PrivilegedStateMsg), 0) != 0)
    {
        // 队列满: 消息被丢弃 (g1_ctrl 读得太慢)
        // 非致命, 下一帧继续
    }
}


// =============================================================================
// cleanup_privileged_publisher — 删除话题 (仿真退出时调用)
// =============================================================================

void cleanup_privileged_publisher()
{
    if (g_mqd >= 0)
    {
        mq_close(g_mqd);
        mq_unlink(TOPIC_NAME);
        g_mqd = -1;
        printf("[PrivilegedPublisher] topic '%s' removed\n", TOPIC_NAME);
    }
}


// =============================================================================
// 集成示例 — unitree_mujoco 仿真主循环:
//
//   int main() {
//       mjModel* m = mj_loadXML("scene.xml", nullptr, ...);
//       mjData* d = mj_makeData(m);
//
//       init_privileged_publisher(m);   // 启动时调用一次
//
//       while (running) {
//           mj_step(m, d);
//
//           // ... 发布 DDS LowState ...
//
//           publish_privileged_state(m, d);  // ★ 每帧发布特权信息
//
//           // ... 订阅 DDS LowCmd ...
//       }
//
//       cleanup_privileged_publisher();
//       mj_deleteData(d);
//       mj_deleteModel(m);
//   }
// =============================================================================
