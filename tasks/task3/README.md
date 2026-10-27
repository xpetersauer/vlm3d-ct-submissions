# task3 Forithmus Docker context

Track slug: `ct-volume-generation`
Image tag: `vlm3d-task3-ctgen:latest`

## Local schema test sequence

```bash
pip install --upgrade forithmus
forithmus login
forithmus init ct-volume-generation
forithmus generate
./build.sh
./forithmus_test.sh
```

## Manual Docker smoke test with mounted weights

```bash
unzip -o ../weights_task3.zip -d ./weights
./local_docker_smoke_test.sh
```

## Export and submit

```bash
./export.sh
./submit_template.sh
```
