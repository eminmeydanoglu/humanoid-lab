// Hardware-free integration probe for the patched robot-side ZMQ manager.
// Run on the Unitree ARM host against a temporary patched SONIC source copy.
#include <cassert>
#include <chrono>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#include <zmq.h>
#include "zmq_manager.hpp"

int main(int argc, char** argv) {
  assert(argc == 2);  // recorded, valid Unitree SONIC v1.1 Protocol v4 packet
  std::ifstream packet_file(argv[1], std::ios::binary);
  assert(packet_file.good());
  const std::string recorded_pose(std::istreambuf_iterator<char>{packet_file}, {});
  assert(recorded_pose.size() == 1604);
  using namespace std::chrono_literals;
  void* context = zmq_ctx_new();
  assert(context != nullptr);
  void* publisher = zmq_socket(context, ZMQ_PUB);
  assert(publisher != nullptr);
  assert(zmq_bind(publisher, "tcp://127.0.0.1:16556") == 0);
  void* status = zmq_socket(context, ZMQ_SUB);
  assert(status != nullptr);
  int conflate = 1;
  assert(zmq_setsockopt(status, ZMQ_CONFLATE, &conflate, sizeof(conflate)) == 0);
  const std::string status_topic = "sonic_status ";
  assert(zmq_setsockopt(status, ZMQ_SUBSCRIBE, status_topic.data(), status_topic.size()) == 0);
  int timeout_ms = 100;
  assert(zmq_setsockopt(status, ZMQ_RCVTIMEO, &timeout_ms, sizeof(timeout_ms)) == 0);
  assert(zmq_connect(status, "tcp://127.0.0.1:5561") == 0);

  {
    ZMQManager manager("127.0.0.1", 16556);
    MotionDataReader motion_reader;
    auto loaded_motion = std::make_shared<MotionSequence>();
    loaded_motion->name = "hardware_free_fixture";
    motion_reader.motions.push_back(loaded_motion);
    std::shared_ptr<const MotionSequence> current_motion = loaded_motion;
    int current_frame = 0;
    OperatorState operator_state;
    bool reinitialize_heading = false;
    DataBuffer<HeadingState> heading_state;
    PlannerState planner_state;
    DataBuffer<MovementState> movement_state;
    std::mutex current_motion_mutex;
    bool report_temperature = false;
    auto process_manager = [&]() {
      manager.update();
      if (manager.GetManagedMode() == ZMQManager::ManagedMode::STREAMED_MOTION) {
        manager.handle_input(motion_reader, current_motion, current_frame,
                             operator_state, reinitialize_heading, heading_state,
                             true, planner_state, movement_state,
                             current_motion_mutex, report_temperature);
      }
    };
    std::this_thread::sleep_for(300ms);  // PUB/SUB subscription handshake
    auto await_status_mode = [&](const std::string& expected) {
      for (int i = 0; i < 12; ++i) {
        process_manager();
        char buffer[512]{};
        int size = zmq_recv(status, buffer, sizeof(buffer), 0);
        if (size > 0) {
          std::string message(buffer, static_cast<size_t>(size));
          if (message.find("\"mode\":\"" + expected + "\"") != std::string::npos) return true;
        }
        std::this_thread::sleep_for(20ms);
      }
      return false;
    };

    std::string header = R"({"v":1,"endian":"le","count":1,"fields":[{"name":"start","dtype":"u8","shape":[1]},{"name":"stop","dtype":"u8","shape":[1]},{"name":"planner","dtype":"u8","shape":[1]}]})";
    assert(header.size() < 1280);
    header.resize(1280, '\0');
    std::string command = "command" + header;
    command.append(3, '\0');  // start=false, stop=false, planner=false

    bool entered_stream = false;
    for (int i = 0; i < 20 && !entered_stream; ++i) {
      assert(zmq_send(publisher, command.data(), command.size(), 0) == static_cast<int>(command.size()));
      process_manager();
      entered_stream = manager.GetManagedMode() == ZMQManager::ManagedMode::STREAMED_MOTION;
      std::this_thread::sleep_for(20ms);
    }
    assert(entered_stream);
    assert(await_status_mode("STREAMED_MOTION"));

    // UI Stop is a mode change to planner, not command.stop=true (which stops
    // the controller). Confirm the normal idle transition before fault testing.
    std::string idle_command = command;
    idle_command.back() = '\1';
    bool explicit_idle = false;
    for (int i = 0; i < 20 && !explicit_idle; ++i) {
      assert(zmq_send(publisher, idle_command.data(), idle_command.size(), 0) ==
             static_cast<int>(idle_command.size()));
      process_manager();
      explicit_idle = manager.GetManagedMode() == ZMQManager::ManagedMode::PLANNER;
      std::this_thread::sleep_for(20ms);
    }
    assert(explicit_idle);
    assert(await_status_mode("PLANNER"));

    entered_stream = false;
    for (int i = 0; i < 20 && !entered_stream; ++i) {
      assert(zmq_send(publisher, command.data(), command.size(), 0) ==
             static_cast<int>(command.size()));
      process_manager();
      entered_stream = manager.GetManagedMode() == ZMQManager::ManagedMode::STREAMED_MOTION;
      std::this_thread::sleep_for(20ms);
    }
    assert(entered_stream);

    // A production-converted Unitree v1.1 frame must refresh the valid-token
    // clock, unlike malformed packets. It is only decoded here; no DDS/motors.
    bool accepted_recorded_pose = false;
    for (int i = 0; i < 15 && !accepted_recorded_pose; ++i) {
      assert(zmq_send(publisher, recorded_pose.data(), recorded_pose.size(), 0) ==
             static_cast<int>(recorded_pose.size()));
      process_manager();
      char buffer[512]{};
      int size = zmq_recv(status, buffer, sizeof(buffer), 0);
      if (size > 0) {
        const std::string message(buffer, static_cast<size_t>(size));
        accepted_recorded_pose = message.find("\"mode\":\"STREAMED_MOTION\"") != std::string::npos &&
            message.find("\"valid_token_age_ms\":null") == std::string::npos;
      }
      std::this_thread::sleep_for(20ms);
    }
    assert(accepted_recorded_pose);

    std::string bad_header = R"({"v":4,"endian":"le","count":1,"fields":[{"name":"token_state","dtype":"f32","shape":[1,1]}]})";
    bad_header.resize(1280, '\0');
    std::string malformed_pose = "pose" + bad_header;
    malformed_pose.append(4, '\0');  // wrong token width

    // Malformed packets continue to arrive, but no valid Protocol v4 token is
    // accepted. The robot-side input manager must
    // return to planner mode on its own, without a workstation stop command.
    bool returned_to_planner = false;
    for (int i = 0; i < 30 && !returned_to_planner; ++i) {
      assert(zmq_send(publisher, malformed_pose.data(), malformed_pose.size(), 0) ==
             static_cast<int>(malformed_pose.size()));
      process_manager();
      returned_to_planner = manager.GetManagedMode() == ZMQManager::ManagedMode::PLANNER;
      std::this_thread::sleep_for(20ms);
    }
    assert(returned_to_planner);
    assert(await_status_mode("PLANNER"));
  }

  zmq_close(status);
  zmq_close(publisher);
  zmq_ctx_term(context);
}
