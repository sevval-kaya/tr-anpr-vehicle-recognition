# Katkı notları

## Bağımlılıkları yeniden kilitleme

`requirements-lock.txt`, `pyproject.toml`'daki `dev`/`detection`/`ocr`/`serving`
gruplarının tamamı için pinlenmiş, [pip-tools](https://github.com/jazzband/pip-tools)
ile üretilmiş bir kilit dosyasıdır (`pip install pip-tools` gerekir).
`pyproject.toml`'daki bağımlılıklar değiştiğinde yeniden üretmek için:

```bash
pip-compile --extra dev --extra detection --extra ocr --extra serving -o requirements-lock.txt
```

> **Not:** Bu dosya şu an Windows + Python 3.13 üzerinde üretildi
> (`pyproject.toml`'ın gerektirdiği minimum 3.11 değil — geliştirme
> makinesinde kurulu olan buydu). Bazı pinler platforma/Python sürümüne
> özel wheel seçebilir; farklı bir işletim sistemi/Python sürümünde tam
> yeniden üretilebilirlik gerekiyorsa (örn. CI'da kilitlenmiş kurulum),
> dosyayı o ortamda yeniden üretmek daha güvenli olur.
