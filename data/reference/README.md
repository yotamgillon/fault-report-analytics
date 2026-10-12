# Reference data

`vendors.csv` is **synthetic, generic vendor master data** used to join vendor information onto
fault reports (tier, SLA target, support region, contract type).

- Tiers, SLA targets, regions and contract types were assigned arbitrarily for this demo. They were
  **not** chosen to make any vendor look good or bad.
- `sla_target_days` is the resolution-time target in days: a resolved ticket that took longer is an
  SLA breach. Stricter targets go with higher tiers (Tier 1 = 5, Tier 2 = 7, Tier 3 = 10).
- Replace this file with real vendor data and re-run `make all`; nothing else needs to change.

`challenge_descriptions.csv` is a small test set for the text classifier, written by the same AI
assistant that wrote the rules, so it is not independent evidence (see the
data-quality report).
