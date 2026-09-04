# ROS 2 / DDS

Ana container varsayilan olarak dogrudan DDS kullanir (network_mode: host); ayri Foxy container'i yok.
Bu klasordeki dosyalar robot agrina asla yazmaz.

Stage A (ana container'dan direct DDS, motor komutu yok):
1. ROS_DOMAIN_ID, RMW vendor, NIC/subnet, topic/service listesini kaydet.
2. Robot NIC'ine host network ile baglan.
3. Multicast/UDP + firewall dogrula; discovery dene.
4. Salt-okunur message echo; custom interface'leri ayni kaynaktan build edip read-only decode et; QoS esle.
5. En son guvenli test publisher (lab kosullari).

Stage B (minimal Foxy servis container'i, sadece gercek ihtiyac halinde):
- digest-pinned Foxy image, CUDA/Isaac/GR00T yok, host network, bind-mounted workspace, sadece ./dev.sh foxy ile.

direct-dds-test.sh: robot erisim parametreleri (NIC/IP, ROS_DOMAIN_ID, RMW, interface repo) belirlenince doldurulacak template.
