# Troubleshooting Guide

Common issues when setting up vLLM on DGX Spark and their solutions.

## Table of Contents
- [Model Loading Issues](#model-loading-issues)
- [Performance Issues](#performance-issues)
- [Container Issues](#container-issues)
- [Auto-Start / Systemd Issues](#auto-start--systemd-issues)
- [Open WebUI Connection Issues](#open-webui-connection-issues)
- [Memory Issues](#memory-issues)

## Model Loading Issues

### "model type gemma4 not recognized"

**Symptom**:
```
Value error, The checkpoint you are trying to load has model type `gemma4` 
but Transformers does not recognize this architecture.
```

**Cause**: The container's built-in `transformers` version is too old for Gemma 4.

**Solution**: This repo handles it automatically. `scripts/start.sh` mounts `scripts/startup.sh` into the container, which runs `pip install --upgrade transformers` before launching vLLM. If you are running Docker manually, make sure you use the `--entrypoint` and startup script approach shown in the README.

### "Cannot find model"

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/huggingface/...'
```

**Cause**: Volume mount incorrect or model not downloaded.

**Solution**:
```bash
# Ensure cache directory exists on host
mkdir -p ~/.cache/huggingface

# Pre-download the model
bash scripts/download-model.sh

# Check the mount path
docker inspect vllm-gemma4-26b | grep -A 5 Mounts
```

### "No such file or directory: 'gemma4_patched.py'"

**Symptom**: `scripts/start.sh` exits immediately with a patch-not-found error.

**Cause**: You may have moved the repo after cloning, or the patch file is missing.

**Solution**: Ensure you run `scripts/start.sh` from inside the cloned repository directory. The script looks for `patches/gemma4_patched.py` relative to itself.

### Slow first startup

**Symptom**: First request takes 5-10 minutes.

**Cause**: This is normal! The container is:
1. Upgrading `transformers` inside the container
2. Downloading the model (~15GB) if not pre-downloaded
3. Loading weights (~100s)
4. Compiling CUDA graphs (~55s)

**Solution**: Pre-download the model:
```bash
bash scripts/download-model.sh
```

## Performance Issues

### Very slow inference (~9 tok/s instead of 45+)

**Symptom**: Getting ~9-10 tok/s instead of expected 45+.

**Likely Cause**: Using wrong model or wrong quantization flag.

**Check**:
```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

**Fixes**:
1. Ensure you're using `compressed-tensors` quantization:
   ```bash
   --quantization compressed-tensors  # ✅ Fast
   # NOT
   --quantization modelopt            # ❌ Slow on DGX Spark
   ```

2. Ensure CUDA graphs are enabled (check logs for "Capturing CUDA graphs")

3. Verify the model is AEON-7's 26B, not LilaRest's 31B

4. Verify the `gemma4_patched.py` is mounted into the container. Without the patch, vLLM may silently fall back to a slower path or error out.

### "Not enough SMs to use max_autotune_gemm mode"

**Symptom**: Warning in logs.

**Cause**: DGX Spark GB10 has fewer SMs than datacenter GPUs.

**Solution**: This is a warning, not an error. Performance is still optimal.

## Container Issues

### "docker: Error response from daemon: could not select device driver"

**Cause**: NVIDIA Container Toolkit not installed.

**Solution**:
```bash
# Install nvidia-docker2 (modern keyring method)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Container exits immediately

**Symptom**: Container starts then exits with no logs.

**Check**:
```bash
docker logs vllm-gemma4-26b
```

**Common Causes**:
1. Port 8000 already in use
2. Out of disk space
3. Permission denied on cache directory
4. Missing `gemma4_patched.py` mount

**Solution**:
```bash
# Check port usage
sudo lsof -i :8000

# Check disk space
df -h

# Fix permissions
sudo chown -R $USER:$USER ~/.cache/huggingface

# Verify patch file exists
ls -la patches/gemma4_patched.py
```

## Auto-Start / Systemd Issues

### "Failed to start vllm-gemma4-26b.service"

**Symptom**: `systemctl --user start vllm-gemma4-26b.service` fails.

**Cause 1 — `docker.service` not found**: On some distros, Docker runs as a root service but user systemd doesn't see `docker.service` as a dependency.

**Solution**: The service file in this repo uses `After=docker.service` without `Requires=docker.service` to avoid this. If you still have issues, start Docker manually:
```bash
sudo systemctl start docker
systemctl --user start vllm-gemma4-26b.service
```

**Cause 2 — `216/GROUP` error**: The `User=` directive in an old version of the service file conflicts with user-mode systemd.

**Solution**: Re-run `scripts/install-service.sh` to get the latest service file, then:
```bash
systemctl --user daemon-reload
systemctl --user restart vllm-gemma4-26b.service
```

### Service doesn't start after reboot

**Cause**: User systemd services only auto-start on login, not on system boot, unless lingering is enabled.

**Solution**:
```bash
# Enable lingering so user services start at boot even without login
sudo loginctl enable-linger $USER
```

## Open WebUI Connection Issues

### "Connection refused" when Open WebUI tries to reach vLLM

**Symptom**: Open WebUI shows an error or no models appear.

**Cause 1**: vLLM is not running.

**Solution**:
```bash
# Verify vLLM is up
curl http://localhost:8000/v1/models

# If not, start it
bash scripts/start.sh
```

**Cause 2**: Open WebUI is running inside Docker and `localhost` / `127.0.0.1` points to the container itself, not the host.

**Solution**: Use the host-resolver DNS name inside the container:
```
http://host.docker.internal:8000/v1
```
If `host.docker.internal` doesn't work on your Linux system, use one of these alternatives:
- Start the Open WebUI container with `--network=host` (Linux only), then `http://localhost:8000/v1` works.
- Use the host's LAN IP address, e.g. `http://192.168.1.42:8000/v1`.

**Cause 3**: Wrong API path.

**Solution**: Ensure the API base URL ends with `/v1`:
```
http://localhost:8000/v1
```

## Memory Issues

### "OutOfMemoryError: CUDA out of memory"

**Symptom**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions**:

1. Reduce `--gpu-memory-utilization`:
   ```bash
   --gpu-memory-utilization 0.50  # Instead of 0.60
   ```

2. Reduce `--max-model-len`:
   ```bash
   --max-model-len 131072  # Instead of 262000
   ```

3. Reduce `--max-num-seqs`:
   ```bash
   --max-num-seqs 64  # Instead of 128
   ```

### System freezes during startup

**Cause**: Compiling CUDA graphs uses significant CPU/RAM.

**Solution**: This is normal for 30-60 seconds. Wait it out.

If it persists >5 minutes, reduce model length or disable CUDA graphs (will hurt performance):
```bash
--enforce-eager  # Disables CUDA graphs
```

## Getting Help

If issues persist:

1. Check vLLM logs:
   ```bash
   docker logs vllm-gemma4-26b --tail 100
   ```

2. Check GPU status:
   ```bash
   nvidia-smi
   ```

3. Open an issue with:
   - Full error message
   - Output of `docker logs`
   - Output of `nvidia-smi`
   - Your Docker run command
