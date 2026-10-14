# memory

Two memory tiers live here: working (`working.py`, Redis `rmsai:session:{sid}`, 24 h TTL renewed on
every save, latest 50 turns kept) and episodic (`episodic.py`, Qdrant `rmsai_episodic`;
Mem0-swappable). The semantic tier is the KB itself (`kb/`, `cli.kb`). Phase 3.
