#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Acts/Plugins/ExaTrkX/BoostTrackBuilding.hpp"
#include "Acts/Plugins/ExaTrkX/ExaTrkXPipeline.hpp"
#include "Acts/Plugins/ExaTrkX/Stages.hpp"
#include "Acts/Plugins/ExaTrkX/TorchEdgeClassifier.hpp"
#include "Acts/Plugins/ExaTrkX/TorchMetricLearning.hpp"
#include "ActsExaTrkX.hpp"
using float32 = float;

std::vector<float32>
readCSVSPs(const std::string& filename)
{
  std::vector<float32> data;
  std::ifstream file(filename);

  if (!file.is_open()) {
    throw std::runtime_error("Error opening file: " + filename);
  }

  std::string line;
  while (std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;

    while (std::getline(lineStream, cell, ',')) {
      try {
        float32 value = std::stof(cell);
        data.push_back(value);
      }
      catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << std::endl;
      }
      catch (const std::out_of_range& e) {
        std::cerr << "Out of range: " << e.what() << std::endl;
      }
    }
  }

  return data;
}

int
main(int argc, char* argv[])
{
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <event*-converted.csv>"
              << " <device_id> <log_level>" << std::endl;
    return 1;
  }

  int device_id;
  int log_level;

  try {
    device_id = std::stoi(argv[2]);
    log_level = std::stoi(argv[3]);
  }
  catch (const std::invalid_argument& e) {
    std::cerr << "Invalid argument: " << e.what() << std::endl;
    return 1;
  }
  catch (const std::out_of_range& e) {
    std::cerr << "Argument out of range: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Running Standalone version of Acts ExaTrkXPipeline..."
            << std::endl;
  std::cout << "Device ID: " << device_id << ", Log Level: " << log_level
            << std::endl;

  std::string modelDir =
      "/global/cfs/projectdirs/m3443/data/ACTS-aaS/models/smeared_hits/";
  auto actsExaTrkX =
      std::make_unique<ActsExaTrkX>(modelDir, device_id, log_level);

  try {
    std::vector<float> features = readCSVSPs(argv[1]);
    int numFeatures = 3;
    if (features.size() % numFeatures != 0) {
      std::cerr << "The number of data points in the CSV file is not a "
                   "multiple of numFeatures."
                << std::endl;
      return 1;
    }

    int numSpacepoints = features.size() / numFeatures;
    std::vector<int> spacepoint_ids(numSpacepoints);
    std::iota(
        spacepoint_ids.begin(), spacepoint_ids.end(), 0);  // 生成spacepoint_ids

    auto track_candidates = actsExaTrkX->runPipeline(features, spacepoint_ids);
    std::cout << "Size of track_candidates: " << track_candidates.size()
              << std::endl;

    for (size_t i = 0; i < track_candidates.size() && i < 10; ++i) {
      std::cout << "track_candidates[" << i
                << "] (size: " << track_candidates[i].size() << "): ";
      for (size_t j = 0; j < track_candidates[i].size() && j < 10; ++j) {
        std::cout << track_candidates[i][j] << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "Done Acts ExaTrkXPipeline!" << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
