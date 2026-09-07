#include <onnxruntime_cxx_api.h>
#include <zmq.hpp>
#include "input_interface/zmq_packed_message_subscriber.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr size_t kBody = 29, kState = 93, kHistory = 10, kToken = 64;
constexpr size_t kDecoderInput = kState * kHistory + kToken;
constexpr std::array<char, 5> kStateMagic = {'C','W','S','T','1'};
constexpr std::array<char, 5> kBodyMagic = {'C','W','B','D','1'};

const char* required_env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') throw std::runtime_error(std::string("missing environment: ") + name);
  return value;
}
uint64_t now_ns() { return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); }
uint64_t read_u64(const char* data) { uint64_t value; std::memcpy(&value, data, sizeof(value)); return value; }
int64_t read_i64(const void* data) { int64_t value; std::memcpy(&value, data, sizeof(value)); return value; }
void write_u64(char* data, uint64_t value) { std::memcpy(data, &value, sizeof(value)); }
void event(const char* name, uint64_t sequence, uint64_t timestamp, uint64_t count) {
  std::cout << "{\"event\":\"" << name << "\",\"sequence\":" << sequence << ",\"monotonic_ns\":" << timestamp << ",\"count\":" << count << "}" << std::endl;
}

class Decoder {
 public:
  explicit Decoder(const char* path) : env_(ORT_LOGGING_LEVEL_WARNING, "cloudwalk-sonic"), session_(env_, path, options_) {
    Ort::AllocatorWithDefaultOptions allocator;
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) throw std::runtime_error("SONIC decoder I/O count changed");
    input_name_ = session_.GetInputNameAllocated(0, allocator).get(); output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
    shape_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape_.size() != 2 || shape_[0] != 1 || shape_[1] != static_cast<int64_t>(kDecoderInput)) throw std::runtime_error("SONIC decoder input is not [1,994]");
  }
  std::array<float, kBody> run(const std::array<float, kDecoderInput>& input) {
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()), input.size(), shape_.data(), shape_.size());
    const char* in = input_name_.c_str(); const char* out = output_name_.c_str(); auto result = session_.Run(Ort::RunOptions{nullptr}, &in, &tensor, 1, &out, 1);
    if (result[0].GetTensorTypeAndShapeInfo().GetElementCount() != kBody) throw std::runtime_error("SONIC decoder output is not 29D");
    const float* values = result[0].GetTensorData<float>();
    if (!std::all_of(values, values + kBody, [](float value) { return std::isfinite(value); })) throw std::runtime_error("SONIC decoder emitted non-finite command");
    std::array<float, kBody> command{}; std::copy(values, values + kBody, command.begin()); return command;
  }
 private:
  Ort::Env env_; Ort::SessionOptions options_; Ort::Session session_; std::string input_name_, output_name_; std::vector<int64_t> shape_;
};
}  // namespace

int main() {
  try {
    const int action_port = std::stoi(required_env("CLOUDWALK_ACTION_PORT"));
    const int state_port = std::stoi(required_env("CLOUDWALK_STATE_PORT"));
    const int body_port = std::stoi(required_env("CLOUDWALK_BODY_PORT"));
    Decoder decoder(required_env("DECODER_MODEL"));
    zmq::context_t context(1);
    zmq::socket_t body_socket(context, zmq::socket_type::pub);
    body_socket.set(zmq::sockopt::linger, 0); body_socket.bind("tcp://127.0.0.1:" + std::to_string(body_port));
    std::array<float, kState * kHistory> history{}; uint64_t state_sequence = 0, state_time = 0, state_count = 0;
    std::mutex state_mutex;
    std::thread state_receiver([&] {
      zmq::socket_t socket(context, zmq::socket_type::sub);
      socket.set(zmq::sockopt::subscribe, ""); socket.set(zmq::sockopt::rcvtimeo, 100); socket.connect("tcp://127.0.0.1:" + std::to_string(state_port));
      for (;;) {
        zmq::message_t packet;
        if (!socket.recv(packet, zmq::recv_flags::none)) continue;
        if (packet.size() != 5 + 8 + 8 + 4 * kState || std::memcmp(packet.data(), kStateMagic.data(), 5) != 0) throw std::runtime_error("Isaac state packet rejected");
        const char* data = static_cast<const char*>(packet.data()); const uint64_t received_at = now_ns();
        std::lock_guard<std::mutex> lock(state_mutex);
        state_sequence = read_u64(data + 5); state_time = received_at; ++state_count;
        std::move(history.begin() + kState, history.end(), history.begin()); std::memcpy(history.data() + kState * (kHistory - 1), data + 21, kState * sizeof(float));
        event("native_state_receive", state_sequence, received_at, state_count);
      }
    });
    state_receiver.detach();
    std::atomic<uint64_t> action_count{0}, decoder_count{0}, body_count{0};
    ZMQPackedMessageSubscriber actions("127.0.0.1", action_port, "pose", 20, false, true, 1);
    actions.SetOnDecodedMessage([&](const std::string& topic, const ZMQPackedMessageSubscriber::DecodedHeader& header, const std::vector<ZMQPackedMessageSubscriber::BufferView>& fields) {
      if (topic != "pose" || header.version != 4 || header.endian != "le" || fields.size() != 4 || fields[0].size != 256 || fields[1].size != 8 || fields[2].size != 28 || fields[3].size != 28) throw std::runtime_error("upstream protocol-v4 action rejected");
      const int64_t frame_index = read_i64(fields[1].data); if (frame_index < 0) throw std::runtime_error("negative protocol-v4 frame index");
      const uint64_t sequence = static_cast<uint64_t>(frame_index), received_at = now_ns(), received_count = ++action_count;
      event("native_action_receive", sequence, received_at, received_count);
      std::array<float, kDecoderInput> input{};
      { std::lock_guard<std::mutex> lock(state_mutex);
        if (state_count < kHistory || received_at - state_time > 100000000ULL) { event("native_action_hold", sequence, received_at, state_count); return; }
        std::copy(history.begin(), history.end(), input.begin()); }
      std::memcpy(input.data() + kState * kHistory, fields[0].data, kToken * sizeof(float));
      const uint64_t started = now_ns(); event("native_decoder_invoke", sequence, started, ++decoder_count);
      const auto command = decoder.run(input);
      std::array<char, 5 + 8 + 8 + 4 * kBody> packet{};
      std::copy(kBodyMagic.begin(), kBodyMagic.end(), packet.begin()); write_u64(packet.data() + 5, sequence); write_u64(packet.data() + 13, now_ns());
      std::memcpy(packet.data() + 21, command.data(), kBody * sizeof(float)); body_socket.send(zmq::buffer(packet), zmq::send_flags::none);
      event("native_body_publish", sequence, now_ns(), ++body_count);
    });
    if (!actions.Connect()) throw std::runtime_error("upstream SONIC action subscriber connection failed");
    std::cout << "{\"event\":\"native_ready\",\"decoder_input\":994,\"decoder_output\":29,\"protocol\":\"v4\"}" << std::endl;
    for (;;) actions.PollOnce();
  } catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << std::endl; return 1; }
}
