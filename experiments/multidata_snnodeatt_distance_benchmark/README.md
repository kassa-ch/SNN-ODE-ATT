# Multidata SNNODEATT Distance Benchmark

This directory contains the minimal Phase 1 adapters for the multidataset
SNNODEATT distance-scoring mainline.

Scope:

- normalize legacy and canonical masks into sample-level `mask [B,T]`;
- load processed BTD payloads or synthetic BTD fallback data;
- extract per-time SNNODEATT hidden trajectories;
- call existing score methods through a no-leak wrapper;
- smoke-test online prefix forward without using full-sequence hidden slices.

This directory intentionally does not launch formal training and does not add
new distance mathematics.
