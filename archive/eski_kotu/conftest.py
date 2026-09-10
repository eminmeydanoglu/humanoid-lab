"""Arşivlenmiş eski testler yeni hattın pytest koşusuna katılmaz.

Bu dosya kök yapılandırmaya dokunmadan yalnız bu alt ağacı koleksiyondan
çıkarır; çıplak `pytest` hâlâ `tests/` altındaki güncel paketi toplar.
"""

collect_ignore_glob = ["tests/*"]
