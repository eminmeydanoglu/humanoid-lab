# Çalışma ortamı

Bu checkout bulunan bilgisayar yalnızca kod yazma, kod inceleme ve görsel erişim için kullanılır. Docker, GPU, Isaac Sim/Isaac Lab, MuJoCo, SONIC, GR00T, model indirme, eğitim, değerlendirme ve çalışma zamanı doğrulamaları burada çalıştırılmaz.

Kurulum ve tüm çalışma zamanı işlemleri workhorse olan Raider üzerinde `ssh raider` aracılığıyla yürütülür. Raider üzerindeki ilgili checkout'a geçtikten sonra `setup.sh`, `dev.sh`, `doctor.sh`, `scripts/smoke-test.sh` ve GPU/Docker komutlarını orada çalıştır.

Bu makinede yalnızca kaynak değişiklikleri ile statik kontroller yapılabilir. Raider üzerinde çalıştırılmadan GPU, container veya uçtan uca davranış doğrulanmış olarak bildirilmez.
