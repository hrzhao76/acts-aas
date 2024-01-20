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
  std::cout << "Hello World!" << std::endl;

  std::string modelDir =
      "/global/cfs/projectdirs/m3443/data/ACTS-aaS/models/smeared_hits/";
  std::string metricLearningmodelPath = modelDir + "embed.pt";
  std::string filtermodelPath = modelDir + "filter.pt";
  std::string gnnmodelPath = modelDir + "gnn.pt";

  auto metricLearningLogger =
      Acts::getDefaultLogger("MetricLearning", Acts::Logging::VERBOSE);
  auto filterLogger =
      Acts::getDefaultLogger("FilterModel", Acts::Logging::VERBOSE);
  auto gnnLogger = Acts::getDefaultLogger("GNNModel", Acts::Logging::VERBOSE);
  auto trackBuilderLogger =
      Acts::getDefaultLogger("TrackBuilder", Acts::Logging::VERBOSE);
  int numFeatures = 3;
  Acts::TorchMetricLearning::Config metricLearningConfig;
  Acts::TorchEdgeClassifier::Config filterConfig;
  Acts::TorchEdgeClassifier::Config gnnConfig;
  std::vector<std::shared_ptr<Acts::EdgeClassificationBase>> edgeClassifiers;

  metricLearningConfig.modelPath = metricLearningmodelPath;
  metricLearningConfig.numFeatures = 3;
  metricLearningConfig.embeddingDim = 12;
  std::shared_ptr<Acts::GraphConstructionBase> graphConstructor =
      std::make_shared<Acts::TorchMetricLearning>(
          metricLearningConfig, std::move(metricLearningLogger));

  // Set up the edge classifiers
  filterConfig.modelPath = filtermodelPath;
  filterConfig.numFeatures = 3;
  auto filterClassifier = std::make_shared<Acts::TorchEdgeClassifier>(
      filterConfig, std::move(filterLogger));

  gnnConfig.modelPath = gnnmodelPath;
  gnnConfig.numFeatures = numFeatures;
  auto gnnClassifier = std::make_shared<Acts::TorchEdgeClassifier>(
      gnnConfig, std::move(gnnLogger));

  edgeClassifiers = {filterClassifier, gnnClassifier};

  // Set up the track builder
  auto trackBuilder =
      std::make_shared<Acts::BoostTrackBuilding>(std::move(trackBuilderLogger));

  Acts::ExaTrkXPipeline pipeline(
      graphConstructor, edgeClassifiers, trackBuilder,
      Acts::getDefaultLogger("ExaTrkXPipeline", Acts::Logging::VERBOSE));


  try {
    std::vector<float32> features = readCSVSPs(argv[1]);
    if (features.size() % numFeatures != 0) {
      std::cerr << "The number of data points in the CSV file is not a "
                   "multiple of numFeatures."
                << std::endl;
      return 1;
    }

    int numSpacepoints = features.size() / numFeatures;
    std::vector<int> spacepoint_ids;
    for (int i = 0; i < numSpacepoints; ++i) {
      spacepoint_ids.push_back(i);
    }

    // Run the pipeline
    std::vector<std::vector<int>> track_candidates =
        pipeline.run(features, spacepoint_ids);

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
  }
  catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }


  return 0;
}
