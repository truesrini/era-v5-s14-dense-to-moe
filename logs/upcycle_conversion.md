| method | val loss | change | total params | active params |
|---|---|---|---|---|
| dense model (before conversion) | 1.9920 | +0.0000 | 8.0M | 8.0M |
| copy: 8 full copies, top-2 (sparse upcycling) | 1.9920 | +0.0000 | 41.0M | 12.7M |
| partition, random neurons -> shared, random subsets -> experts | 2.0824 | +0.0904 | 24.5M | 8.0M |
| partition, important neurons -> shared (ours) | 2.0515 | +0.0595 | 24.5M | 8.0M |
| partition (ours) + drop-upcycling r=0.5 | 2.1438 | +0.1518 | 24.5M | 8.0M |
| partition (ours), route_scale 1 instead of 2 | 2.0908 | +0.0988 | 24.5M | 8.0M |
| no shared expert: 16 experts x 512, top-2 | 2.1897 | +0.1977 | 41.1M | 8.0M |
| scratch: attention copied, FFN re-initialised | 10.0132 | +8.0211 | 24.5M | 8.0M |
