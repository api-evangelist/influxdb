---
title: "What’s New in InfluxDB 3.12: Distributed Compaction, 3x Faster Imports, and More Control"
url: "https://www.influxdata.com/blog/influxdb-3-12/"
date: "2026-10-01"
author: "Peter Barnett (InfluxData)"
feed_url: "https://www.influxdata.com/blog/feed/"
---
Time series deployments rarely grow the way they were planned. A single-node pilot becomes a multi-node fleet, an afternoon’s worth of test data becomes weeks of backfill that has to land before anyone can query it, and three teams end up writing to a database built for one. InfluxDB 3.11 delivered a higher-performance foundation for complex time series workloads, with significant performance improvements for single-series queries, schemas that could go wide and sparse without breaking, and a much more predictable resource profile.
