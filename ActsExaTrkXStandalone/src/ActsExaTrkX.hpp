#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Acts/Plugins/ExaTrkX/BoostTrackBuilding.hpp"
#include "Acts/Plugins/ExaTrkX/ExaTrkXPipeline.hpp"
#include "Acts/Plugins/ExaTrkX/Stages.hpp"
#include "Acts/Plugins/ExaTrkX/TorchEdgeClassifier.hpp"
#include "Acts/Plugins/ExaTrkX/TorchMetricLearning.hpp"
#include "Acts/Utilities/Logger.hpp"

class ActsExaTrkX {
 private:
  std::string modelDir_;
  int deviceID_;
  int logLevel_;

  std::unique_ptr<Acts::ExaTrkXPipeline> pipeline_;
  std::shared_ptr<Acts::GraphConstructionBase> graphConstructor_;
  std::vector<std::shared_ptr<Acts::EdgeClassificationBase>> edgeClassifiers_;
  std::shared_ptr<Acts::TrackBuildingBase> trackBuilder_;

 public:
  ActsExaTrkX(const std::string& modelDir, int deviceID, int logLevel);
  ~ActsExaTrkX();

  void initializePipeline();
  std::vector<std::vector<int>> runPipeline(
      std::vector<float>& features, std::vector<int>& spacepoint_ids);
};

ActsExaTrkX::ActsExaTrkX(
    const std::string& modelDir, int deviceID, int logLevel)
    : modelDir_(modelDir), deviceID_(deviceID), logLevel_(logLevel)
{
  initializePipeline();
}

void
ActsExaTrkX::initializePipeline()
{
  // Check if the model directory exists
  if (!std::filesystem::exists(modelDir_)) {
    std::cerr << "The model directory does not exist: " << modelDir_
              << std::endl;
    return;
  }

  std::string metricLearningmodelPath = modelDir_ + "/embed.pt";
  std::string filtermodelPath = modelDir_ + "/filter.pt";
  std::string gnnmodelPath = modelDir_ + "/gnn.pt";

  // Check if the log level is valid
  if (logLevel_ < 0 ||
      logLevel_ > static_cast<int>(Acts::Logging::Level::MAX)) {
    std::cerr << "The log level is not valid: " << logLevel_ << std::endl;
    return;
  }

  auto metricLearningLogger = Acts::getDefaultLogger(
      "MetricLearning", static_cast<Acts::Logging::Level>(logLevel_));
  auto filterLogger = Acts::getDefaultLogger(
      "FilterModel", static_cast<Acts::Logging::Level>(logLevel_));
  auto gnnLogger = Acts::getDefaultLogger(
      "GNNModel", static_cast<Acts::Logging::Level>(logLevel_));
  auto trackBuilderLogger = Acts::getDefaultLogger(
      "TrackBuilder", static_cast<Acts::Logging::Level>(logLevel_));

  Acts::TorchMetricLearning::Config metricLearningConfig;
  metricLearningConfig.modelPath = metricLearningmodelPath;
  metricLearningConfig.numFeatures =
      3;  // Consider defining as a constant or a class member
  metricLearningConfig.embeddingDim = 8;
  metricLearningConfig.rVal = 0.2;
  metricLearningConfig.knnVal = 100;
  metricLearningConfig.deviceID = deviceID_;

  graphConstructor_ = std::make_shared<Acts::TorchMetricLearning>(
      metricLearningConfig, std::move(metricLearningLogger));

  // Set up the edge classifiers
  Acts::TorchEdgeClassifier::Config filterConfig;
  filterConfig.modelPath = filtermodelPath;
  filterConfig.cut = 0.01;
  filterConfig.nChunks = 5;
  filterConfig.deviceID = deviceID_;

  auto filterClassifier = std::make_shared<Acts::TorchEdgeClassifier>(
      filterConfig, std::move(filterLogger));

  Acts::TorchEdgeClassifier::Config gnnConfig;
  gnnConfig.modelPath = gnnmodelPath;
  gnnConfig.cut = 0.5;
  gnnConfig.undirected = true;
  gnnConfig.deviceID = deviceID_;

  auto gnnClassifier = std::make_shared<Acts::TorchEdgeClassifier>(
      gnnConfig, std::move(gnnLogger));

  edgeClassifiers_ = {filterClassifier, gnnClassifier};

  // Set up the track builder
  trackBuilder_ =
      std::make_shared<Acts::BoostTrackBuilding>(std::move(trackBuilderLogger));

  pipeline_ = std::make_unique<Acts::ExaTrkXPipeline>(
      graphConstructor_, edgeClassifiers_, trackBuilder_,
      Acts::getDefaultLogger(
          "ExaTrkXPipeline", static_cast<Acts::Logging::Level>(logLevel_)));
}

ActsExaTrkX::~ActsExaTrkX()
{
  // Destructor implementation (if needed)
}

std::vector<std::vector<int>>
ActsExaTrkX::runPipeline(
    std::vector<float>& features, std::vector<int>& spacepoint_ids)
{
  // Implementation of runPipeline
  // This function should return the result of running the pipeline
  return pipeline_->run(features, spacepoint_ids);
}
