#include <onnxruntime_cxx_api.h>
#include "input_interface/zmq_packed_message_subscriber.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr std::array<float, 64> kReference = {
    -0.0625f, 0.f, -0.0625f, -0.125f, -0.1875f, -0.0625f, 0.1875f, 0.25f,
    0.1875f, -0.125f, 0.0625f, -0.0625f, -0.25f, -0.25f, -0.3125f, -0.0625f,
    0.f, -0.0625f, -0.125f, -0.1875f, 0.f, -0.25f, 0.f, -0.25f,
    -0.0625f, 0.0625f, 0.125f, -0.125f, 0.25f, 0.1875f, 0.25f, -0.125f,
    0.125f, 0.1875f, -0.0625f, 0.f, -0.1875f, -0.1875f, 0.25f, 0.f,
    0.f, -0.125f, 0.0625f, 0.f, -0.0625f, -0.0625f, 0.1875f, -0.0625f,
    0.f, 0.0625f, 0.125f, 0.0625f, 0.125f, 0.0625f, 0.125f, 0.f,
    0.125f, 0.1875f, 0.f, 0.f, 0.0625f, 0.0625f, 0.1875f, 0.0625f};

const char* required_env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') throw std::runtime_error(std::string("missing environment: ") + name);
  return value;
}

size_t element_count(const std::vector<int64_t>& shape) {
  size_t total = 1;
  for (const auto dimension : shape) {
    if (dimension <= 0) throw std::runtime_error("model has an unsupported dynamic dimension");
    total *= static_cast<size_t>(dimension);
  }
  return total;
}

void run_model(Ort::Env& env, const char* path, size_t expected_input, size_t expected_output, const std::vector<float>& input, const char* label) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  Ort::Session session(env, path, options);
  Ort::AllocatorWithDefaultOptions allocator;
  if (session.GetInputCount() != 1 || session.GetOutputCount() < 1) throw std::runtime_error(std::string(label) + " I/O count changed");
  const auto input_name = session.GetInputNameAllocated(0, allocator);
  const auto output_name = session.GetOutputNameAllocated(0, allocator);
  const auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (element_count(shape) != expected_input || input.size() != expected_input) throw std::runtime_error(std::string(label) + " input contract changed");
  const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()), input.size(), shape.data(), shape.size());
  const char* input_name_ptr = input_name.get();
  const char* output_name_ptr = output_name.get();
  auto output = session.Run(Ort::RunOptions{nullptr}, &input_name_ptr, &tensor, 1, &output_name_ptr, 1);
  const size_t outputs = element_count(output[0].GetTensorTypeAndShapeInfo().GetShape());
  const float* values = output[0].GetTensorData<float>();
  if (outputs != expected_output || !std::all_of(values, values + outputs, [](float value) { return std::isfinite(value); })) throw std::runtime_error(std::string(label) + " output contract changed");
  std::cout << "MODEL " << label << " input=" << expected_input << " output=" << outputs << std::endl;
}
}  // namespace

int main() {
  try {
    if (std::all_of(kReference.begin(), kReference.end(), [](float value) { return value == 0.f; })) throw std::runtime_error("all-zero hold reference rejected");
    constexpr int port = 55991;
    bool valid_packet = false;
    ZMQPackedMessageSubscriber subscriber("127.0.0.1", port, "pose", 1000, false);
    subscriber.SetOnDecodedMessage([&](const std::string& topic, const ZMQPackedMessageSubscriber::DecodedHeader& header, const std::vector<ZMQPackedMessageSubscriber::BufferView>& fields) {
      const std::array<std::string, 4> names = {"token_state", "frame_index", "left_hand_joints", "right_hand_joints"};
      const std::array<size_t, 4> sizes = {256, 8, 28, 28};
      if (topic != "pose" || header.version != 4 || header.endian != "le" || fields.size() != names.size()) throw std::runtime_error("protocol-v4 header rejected");
      for (size_t index = 0; index < names.size(); ++index) if (header.fields[index].name != names[index] || fields[index].size != sizes[index]) throw std::runtime_error("protocol-v4 field rejected");
      if (*static_cast<const int64_t*>(fields[1].data) != 43 || std::memcmp(fields[0].data, kReference.data(), sizeof(kReference)) != 0) throw std::runtime_error("protocol-v4 payload rejected");
      valid_packet = true;
    });
    if (!subscriber.Connect()) throw std::runtime_error("loopback subscriber connection failed");
    const std::string command = "PYTHONPATH='" + std::string(required_env("PROJECT_ROOT")) + ":" + required_env("SONIC_ROOT") + "' python3 '" + std::string(required_env("PROJECT_ROOT")) + "/scripts/sonic-isolated-vla-producer.py' --port 55991 --sonic-root '" + required_env("SONIC_ROOT") + "' --reference '" + required_env("SAFE_REFERENCE") + "' --decoder '" + required_env("DECODER_MODEL") + "'";
    auto producer = std::async(std::launch::async, [&command] { return std::system(command.c_str()); });
    for (int attempt = 0; attempt < 4 && !valid_packet; ++attempt) subscriber.PollOnce();
    if (producer.get() != 0 || !valid_packet) throw std::runtime_error("upstream VLA producer did not provide a valid packet");
    std::cout << "PROTOCOL v4 upstream_vla_serializer 64+7+7 verified" << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sonic-isolated");
    run_model(env, required_env("ENCODER_MODEL"), 1751, 64, std::vector<float>(1751, 0.001f), "encoder");
    std::vector<float> decoder_input(994, 0.001f);
    std::copy(kReference.begin(), kReference.end(), decoder_input.end() - kReference.size());
    run_model(env, required_env("DECODER_MODEL"), 994, 29, decoder_input, "decoder");
    std::cout << "PASS isolated native harness dds=0 motor_transport=0" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}
