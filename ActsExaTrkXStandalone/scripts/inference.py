#!/usr/bin/env python3
import argparse
import json
import math
import os
import pprint
import subprocess
import sys
import time
import warnings
from pathlib import Path

import acts
import acts.examples
import yaml
from acts.examples.odd import getOpenDataDetector
from acts.examples.reconstruction import *
from acts.examples.simulation import *

u = acts.UnitConstants

logger = acts.logging.getLogger("main")


#########################
# Command line handling #
#########################

parser = argparse.ArgumentParser(
    description="Exa.TrkX data generation/reconstruction script"
)
parser.add_argument("events", help="how many events to run", type=int)
parser.add_argument("models", help="where the models are stored", type=str)
parser.add_argument(
    "digi", help="digitization mode", type=str, choices=["truth", "smear"]
)
parser.add_argument(
    "--npu", "-n", help="how many pileup to generate in one event", type=int, default=0
)
parser.add_argument(
    "--embdim", "-e", help="Hyperparameter embedding dim", type=int, default=8
)
parser.add_argument(
    "--verbose", "-v", help="Make ExaTrkX algorithm verbose", action="store_true"
)
parser.add_argument(
    "--output", "-o", help="where to store output data", type=str, default="output"
)
args = vars(parser.parse_args())

assert args["events"] > 0

outputDir = Path(args["output"])
(outputDir / "train_all").mkdir(exist_ok=True, parents=True)

modelDir = Path(args["models"])

assert (modelDir / "embed.pt").exists()
assert (modelDir / "filter.pt").exists()
assert (modelDir / "gnn.pt").exists()


###########################
# Load Open Data Detector #
###########################

# baseDir = Path(os.path.dirname(__file__))
baseDir = Path(os.environ.get("DEMO_PATH"))
acts_path = os.environ.get("ACTS_PATH")

oddDir = Path(acts_path) / "thirdparty/OpenDataDetector"
if not oddDir.exists():
    oddDir = Path.home() / "Documents/acts_project/acts/thirdparty/OpenDataDetector"

oddMaterialMap = oddDir / "data/odd-material-maps.root"
assert oddMaterialMap.exists()

oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector, trackingGeometry, decorators = getOpenDataDetector(
    oddDir, mdecorator=oddMaterialDeco
)

geoSelectionExaTrkX = baseDir / "detector/odd-geo-selection-whole-detector.json"
assert geoSelectionExaTrkX.exists()

if args["digi"] == "smear":
    digiConfigFile = baseDir / "detector/odd-digi-smearing-config.json"
elif args["digi"] == "truth":
    digiConfigFile = baseDir / "detector/odd-digi-true-config.json"
assert digiConfigFile.exists()


#######################
# Start GPU profiling #
#######################

gpu_profiler_args = [
    "nvidia-smi",
    "--query-gpu=timestamp,index,memory.total,memory.reserved,memory.free,memory.used",
    "--format=csv,nounits",
    "--loop-ms=10",
    "--filename={}".format(outputDir / "gpu_memory_profile.csv"),
]

gpu_profiler = subprocess.Popen(gpu_profiler_args)

#####################
# Prepare sequencer #
#####################

field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=args["events"],
    numThreads=1,
    outputDir=str(outputDir),
    trackFpes=False,
)


#############################
# Simulation & Digitization #
#############################

s = addPythia8(
    s,
    rnd=rnd,
    # hardProcess=["HardQCD:all = on"],
    npileup=args["npu"],
    hardProcess=["Top:qqbar2ttbar=on"],
    outputDirRoot=str(outputDir),
)

particleSelection = ParticleSelectorConfig(
    rho=(0.0 * u.mm, 2.0 * u.mm),
    pt=(500 * u.MeV, 20 * u.GeV),
    absEta=(0, 3),
    removeNeutral=True,
)

addFatras(
    s,
    trackingGeometry,
    field,
    rnd=rnd,
    preSelectParticles=particleSelection,
    # postSelectParticles=particleSelection,
    outputDirRoot=str(outputDir),
)

s = addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfigFile,
    outputDirRoot=None,
    outputDirCsv=str(outputDir / "train_all"),
    rnd=rnd,
    logLevel=acts.logging.INFO,
)

s.addWriter(
    acts.examples.CsvSimHitWriter(
        level=acts.logging.INFO,
        inputSimHits="simhits",
        outputDir=str(outputDir / "train_all"),
        outputStem="truth",
    )
)

s.addWriter(
    acts.examples.CsvMeasurementWriter(
        level=acts.logging.INFO,
        inputMeasurements="measurements",
        inputClusters="clusters",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        outputDir=str(outputDir / "train_all"),
    )
)

s.addWriter(
    acts.examples.CsvTrackingGeometryWriter(
        level=acts.logging.INFO,
        trackingGeometry=trackingGeometry,
        outputDir=str(outputDir),
        writePerEvent=False,
    )
)

#########################
# ExaTrkX Track Finding #
#########################

s.addAlgorithm(
    acts.examples.SpacePointMaker(
        level=acts.logging.INFO,
        inputSourceLinks="sourcelinks",
        inputMeasurements="measurements",
        outputSpacePoints="exatrkx_spacepoints",
        trackingGeometry=trackingGeometry,
        geometrySelection=acts.examples.readJsonGeometryList(str(geoSelectionExaTrkX)),
    )
)

s.addWriter(
    acts.examples.CsvSpacepointWriter(
        level=acts.logging.VERBOSE,
        inputSpacepoints="exatrkx_spacepoints",
        outputDir=str(outputDir / "train_all"),
    )
)

exatrkxLogLevel = acts.logging.VERBOSE if args["verbose"] else acts.logging.INFO

# metricLearningConfig = {
#     # "level": exatrkxLogLevel,
#     "modelPath": str(modelDir / "embed.pt"),
#     "spacepointFeatures": 3,
#     "embeddingDim": args["embdim"],
#     "rVal": 0.2,
#     "knnVal": 100,
# }
metricLearningConfig = {
    "modelPath": str(modelDir / "embed.pt"),
    "embeddingDim": args["embdim"],
    "rVal": 0.2,
    "knnVal": 100,
    "numFeatures": 3,
}

filterConfig = {
    "level": exatrkxLogLevel,
    "cut": 0.01,
    "modelPath": str(modelDir / "filter.pt"),
    "nChunks": 5,
}

gnnConfig = {
    "level": exatrkxLogLevel,
    "cut": 0.5,
    "modelPath": str(modelDir / "gnn.pt"),
    "undirected": True,
}

for cfg in [metricLearningConfig, filterConfig, gnnConfig]:
    assert Path(cfg["modelPath"]).exists()

# graphConstructor = acts.examples.TorchMetricLearning(**metricLearningConfig)
graphConstructor = acts.examples.TorchMetricLearning(
    **metricLearningConfig, level=exatrkxLogLevel
)
edgeClassifiers = [
    acts.examples.TorchEdgeClassifier(**filterConfig),
    acts.examples.TorchEdgeClassifier(**gnnConfig),
]
trackBuilder = acts.examples.BoostTrackBuilding(level=acts.logging.VERBOSE)

s.addAlgorithm(
    acts.examples.TrackFindingAlgorithmExaTrkX(
        level=exatrkxLogLevel,
        inputSpacePoints="exatrkx_spacepoints",
        outputProtoTracks="exatrkx_prototracks",
        graphConstructor=graphConstructor,
        edgeClassifiers=edgeClassifiers,
        trackBuilder=trackBuilder,
        rScale=1000.0,
        phiScale=3.14,
        zScale=1000.0,
    )
)

s.addWriter(
    acts.examples.TrackFinderPerformanceWriter(
        level=acts.logging.INFO,
        inputProtoTracks="exatrkx_prototracks",
        inputParticles="particles_initial",
        inputMeasurementParticlesMap="measurement_particles_map",
        filePath=str(outputDir / "track_finding_performance_exatrkx.root"),
    )
)


#################
# Track fitting #
#################
s.addAlgorithm(
    acts.examples.PrototracksToSeeds(
        level=acts.logging.INFO,
        inputSpacePoints="exatrkx_spacepoints",
        inputProtoTracks="exatrkx_prototracks",
        outputSeeds="exatrkx_seeds",
        outputProtoTracks="exatrkx_prototracks_after_seeds",
    )
)

s.addAlgorithm(
    acts.examples.TrackParamsEstimationAlgorithm(
        level=acts.logging.INFO,
        inputSeeds="exatrkx_seeds",
        outputTrackParameters="exatrkx_estimated_parameters",
        trackingGeometry=trackingGeometry,
        magneticField=field,
    )
)

kalmanOptions = {
    "multipleScattering": True,
    "energyLoss": True,
    "reverseFilteringMomThreshold": 0.0,
    "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
    "level": acts.logging.INFO,
}

s.addAlgorithm(
    acts.examples.TrackFittingAlgorithm(
        level=acts.logging.INFO,
        calibrator=acts.examples.makePassThroughCalibrator(),
        inputMeasurements="measurements",
        inputSourceLinks="sourcelinks",
        inputProtoTracks="exatrkx_prototracks_after_seeds",
        inputInitialTrackParameters="exatrkx_estimated_parameters",
        outputTracks="exatrkx_kalman_tracks",
        pickTrack=-1,
        fit=acts.examples.makeKalmanFitterFunction(
            trackingGeometry, field, **kalmanOptions
        ),
    )
)


s.run()

# stop GPU profiler
gpu_profiler.kill()
