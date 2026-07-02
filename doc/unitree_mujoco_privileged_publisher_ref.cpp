// =============================================================================
// unitree_mujoco 侧 — 特权信息 DDS Publisher 参考实现
//
// 话题: rt/privileged_state (Cyclone DDS, 域 0)
// 类型: PrivilegedStateMsg (128 bytes POD)
//
// 与 g1_ctrl 的 PrivilegedSubscriber 配套使用。
// 双方在同一个 DDS 域自动发现、自动匹配。
//
// 集成方式: 将此代码合并到 unitree_mujoco/simulate/ 仿真主循环中。
// =============================================================================

#include <dds/dds.hpp>           // Cyclone DDS C++11 API
#include <mujoco/mujoco.h>       // MJCF (根据实际路径调整)
#include <cstring>
#include <cstdio>

// =============================================================================
// PrivilegedStateMsg — 与 g1_ctrl 侧 struct 完全一致 (128 bytes)
// =============================================================================

struct PrivilegedStateMsg
{
    float end_effector_pos_w[15];  // 5 bodies × 3 (60 bytes)
    float root_pos_w[3];           // root link pos (12 bytes)
    float box_pos_w[3];            // box pos (12 bytes)
    float box_quat_w[4];           // box quat w,x,y,z (16 bytes)
    float box_size[3];             // box half-size (12 bytes)
    float goal_pos_w[3];           // goal pos (12 bytes)
    uint32_t sequence;             // frame counter (4 bytes)
    uint32_t padding;              // alignment (4 bytes)
};
static_assert(sizeof(PrivilegedStateMsg) == 128, "Must be 128 bytes");


// =============================================================================
// 全局 DDS 状态
// =============================================================================

static std::unique_ptr<dds::domain::DomainParticipant> g_dp;
static std::unique_ptr<dds::topic::Topic<PrivilegedStateMsg>> g_topic;
static std::unique_ptr<dds::pub::Publisher> g_pub;
static std::unique_ptr<dds::pub::DataWriter<PrivilegedStateMsg>> g_writer;
static uint32_t g_sequence = 0;

// 缓存的 body/site ID
static int g_body_left_palm   = -1;
static int g_body_right_palm  = -1;
static int g_body_left_ankle  = -1;
static int g_body_right_ankle = -1;
static int g_body_head        = -1;
static int g_body_root        = -1;
static int g_body_box         = -1;
static int g_site_goal        = -1;
static int g_geom_box         = -1;

static constexpr const char* TOPIC_NAME = "rt/privileged_state";


// =============================================================================
// init_privileged_publisher — 初始化 DDS DataWriter (仿真启动时调用一次)
// =============================================================================

bool init_privileged_publisher(mjModel* m)
{
    // ---- 1. 缓存 MuJoCo body/site/geom ID ----
    g_body_left_palm   = mj_name2id(m, mjOBJ_BODY, "left_palm_link");
    g_body_right_palm  = mj_name2id(m, mjOBJ_BODY, "right_palm_link");
    g_body_left_ankle  = mj_name2id(m, mjOBJ_BODY, "left_ankle_pitch_link");
    g_body_right_ankle = mj_name2id(m, mjOBJ_BODY, "right_ankle_pitch_link");
    g_body_head        = mj_name2id(m, mjOBJ_BODY, "d455_link");
    g_body_root        = mj_name2id(m, mjOBJ_BODY, "torso_link");
    g_body_box         = mj_name2id(m, mjOBJ_BODY, "box");
    g_site_goal        = mj_name2id(m, mjOBJ_SITE, "goal");

    if (g_body_box >= 0)
    {
        int geom_start = m->body_geomadr[g_body_box];
        int geom_num   = m->body_geomnum[g_body_box];
        if (geom_num > 0 && m->geom_type[geom_start] == mjGEOM_BOX)
            g_geom_box = geom_start;
    }

    // ---- 2. 创建 DDS Participant (域 0, 与 unitree_sdk2 同一域) ----
    try
    {
        g_dp = std::make_unique<dds::domain::DomainParticipant>(0);

        // 创建话题 (Cyclone DDS 自动处理 POD 类型)
        g_topic = std::make_unique<dds::topic::Topic<PrivilegedStateMsg>>(
            *g_dp, TOPIC_NAME);

        // 创建 Publisher
        g_pub = std::make_unique<dds::pub::Publisher>(*g_dp);

        // DataWriter QoS: Reliable + KeepLast(10)
        dds::pub::qos::DataWriterQos dw_qos;
        dw_qos << dds::core::policy::Reliability::Reliable();
        dw_qos << dds::core::policy::Durability::TransientLocal();
        dw_qos << dds::core::policy::History::KeepLast(10);

        g_writer = std::make_unique<dds::pub::DataWriter<PrivilegedStateMsg>>(
            *g_pub, *g_topic, dw_qos);

        printf("[PrivilegedPublisher] DDS topic '%s' ready (domain 0)\n",
               TOPIC_NAME);
        return true;
    }
    catch (const dds::core::Exception& e)
    {
        printf("[PrivilegedPublisher] DDS init failed: %s\n", e.what());
        return false;
    }
}


// =============================================================================
// publish_privileged_state — 发布一帧特权信息 (每个仿真步调用)
// =============================================================================

void publish_privileged_state(mjModel* m, mjData* d)
{
    if (!g_writer) return;

    PrivilegedStateMsg msg;
    std::memset(&msg, 0, sizeof(msg));
    msg.sequence = g_sequence++;

    // ---- (A) 末端执行器世界坐标 ----
    auto copy_body = [&](int id, float* dst) {
        if (id >= 0) {
            dst[0] = d->xpos[id*3+0];
            dst[1] = d->xpos[id*3+1];
            dst[2] = d->xpos[id*3+2];
        }
    };
    copy_body(g_body_left_palm,   msg.end_effector_pos_w + 0);
    copy_body(g_body_right_palm,  msg.end_effector_pos_w + 3);
    copy_body(g_body_left_ankle,  msg.end_effector_pos_w + 6);
    copy_body(g_body_right_ankle, msg.end_effector_pos_w + 9);
    copy_body(g_body_head,        msg.end_effector_pos_w + 12);

    // ---- (B) 根Link位置 ----
    copy_body(g_body_root, msg.root_pos_w);

    // ---- (C) 箱子位置和姿态 ----
    if (g_body_box >= 0)
    {
        msg.box_pos_w[0] = d->xpos[g_body_box*3+0];
        msg.box_pos_w[1] = d->xpos[g_body_box*3+1];
        msg.box_pos_w[2] = d->xpos[g_body_box*3+2];

        // MuJoCo xquat: [w, x, y, z]
        msg.box_quat_w[0] = d->xquat[g_body_box*4+0];
        msg.box_quat_w[1] = d->xquat[g_body_box*4+1];
        msg.box_quat_w[2] = d->xquat[g_body_box*4+2];
        msg.box_quat_w[3] = d->xquat[g_body_box*4+3];
    }

    // ---- (D) 箱子尺寸 (从 mjModel 读取, 静态) ----
    if (g_geom_box >= 0)
    {
        msg.box_size[0] = m->geom_size[g_geom_box*3+0];
        msg.box_size[1] = m->geom_size[g_geom_box*3+1];
        msg.box_size[2] = m->geom_size[g_geom_box*3+2];
    }

    // ---- (E) 目标点 ----
    if (g_site_goal >= 0)
    {
        msg.goal_pos_w[0] = d->site_xpos[g_site_goal*3+0];
        msg.goal_pos_w[1] = d->site_xpos[g_site_goal*3+1];
        msg.goal_pos_w[2] = d->site_xpos[g_site_goal*3+2];
    }

    // ---- (F) 发布到 DDS 话题 ----
    try
    {
        g_writer->write(msg);
    }
    catch (const dds::core::Exception& e)
    {
        // 写入失败 (如 QoS 不匹配), 非致命
    }
}


// =============================================================================
// cleanup_privileged_publisher — 清理 DDS 资源 (仿真退出时调用)
// =============================================================================

void cleanup_privileged_publisher()
{
    g_writer.reset();
    g_pub.reset();
    g_topic.reset();
    g_dp.reset();
    printf("[PrivilegedPublisher] DDS resources released\n");
}


// =============================================================================
// 集成示例 — unitree_mujoco 仿真主循环:
//
//   int main() {
//       mjModel* m = mj_loadXML("scene.xml", nullptr, ...);
//       mjData* d = mj_makeData(m);
//
//       // ★ 初始化 DDS 通信 (unitree_sdk2 已在别处调用 ChannelFactory::Init)
//       init_privileged_publisher(m);
//
//       while (running) {
//           mj_step(m, d);
//
//           // ... 现有: 发布 DDS LowState ...
//
//           // ★ 发布特权信息 (与 LowState 在同一 DDS 总线)
//           publish_privileged_state(m, d);
//
//           // ... 现有: 订阅 DDS LowCmd, 写入 mjData->ctrl ...
//       }
//
//       cleanup_privileged_publisher();
//       mj_deleteData(d);
//       mj_deleteModel(m);
//   }
// =============================================================================
