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

parser = argparse.ArgumentParser(description="Exa.TrkX data generation script")

parser.add_argument("events", help="how many events to run", type=int)
parser.add_argument(
    "--output", "-o", help="where to store output data", type=str, default="output"
)
parser.add_argument(
    "--verbose", "-v", help="Make ExaTrkX algorithm verbose", action="store_true"
)

args = vars(parser.parse_args())

assert args["events"] > 0
outputDir = Path(args["output"])
outputDir.mkdir(parents=True, exist_ok=True)

###########################
# Load Open Data Detector #
###########################
baseDir = Path("/global/cfs/projectdirs/m3443/data/ACTS-aaS/scripts/")
acts_path = os.environ.get("ACTS_PATH")

modelDir = Path("/global/cfs/projectdirs/m3443/data/ACTS-aaS/models/smeared_hits/")

oddDir = Path(acts_path) / "thirdparty/OpenDataDetector"
oddMaterialMap = oddDir / "data/odd-material-maps.root"
assert oddMaterialMap.exists()

oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector, trackingGeometry, decorators = getOpenDataDetector(
    oddDir, mdecorator=oddMaterialDeco
)

geoSelectionExaTrkX = baseDir / "detector/odd-geo-selection-whole-detector.json"
assert geoSelectionExaTrkX.exists()

digiConfigFile = baseDir / "detector/odd-digi-smearing-config.json"
assert digiConfigFile.exists()

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
    hardProcess=["Top:qqbar2ttbar=on"],
    npileup=200,
    outputDirRoot=str(outputDir),
    outputDirCsv=str(outputDir),
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
    inputParticles="particles_input",
    outputParticlesInitial="particles_initial",
    outputParticlesFinal="particles_final",
    # outputSimHits = "simhits-addFatras",
    outputDirRoot=str(outputDir),
    outputDirCsv=str(outputDir),
)

s = addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=digiConfigFile,
    outputDirRoot=str(outputDir),
    outputDirCsv=str(outputDir),
    rnd=rnd,
    logLevel=acts.logging.INFO,
)

s.addWriter(
    acts.examples.CsvSimHitWriter(
        level=acts.logging.INFO,
        inputSimHits="simhits",
        outputDir=str(outputDir),
        outputStem="simhits-CsvSimHitWriter",
    )
)

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
    acts.examples.CsvMeasurementWriter(
        level=acts.logging.INFO,
        inputMeasurements="measurements",
        inputClusters="clusters",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        outputDir=str(outputDir),
    )
)

s.addWriter(
    acts.examples.CsvSpacepointWriter(
        level=acts.logging.VERBOSE,
        inputSpacepoints="exatrkx_spacepoints",
        outputDir=str(outputDir),
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

exatrkxLogLevel = acts.logging.INFO

metricLearningConfig = {
    "modelPath": str(modelDir / "embed.pt"),
    "embeddingDim": 8,
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
    acts.examples.CsvProtoTrackWriter(
        level=acts.logging.INFO,
        inputPrototracks="exatrkx_prototracks",
        inputSpacepoints="exatrkx_spacepoints",
        outputDir=str(outputDir),
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


s.run()
