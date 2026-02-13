# Sscmex Example - Nerves Application for Testing SSCMEx

This is a minimal Nerves application to test the SSCMEx NIF on real SG2002 hardware using the [nerves_system_sg2002](https://github.com/fermuch/nerves_system_sg2002) system.

## Prerequisites

1. Nerves installed and configured
2. SG2002 board (reCamera) with SSH access configured
3. nerves_system_sg2002 system available

## WiFi Configuration

Before building, configure your WiFi credentials:

```bash
# Copy the secrets template
cp config/secrets.exs.example config/secrets.exs

# Edit with your WiFi credentials
# Change ssid and psk to your actual values
nano config/secrets.exs
```

**Note:** `config/secrets.exs` is in `.gitignore` and will not be committed.

## Building the Firmware

```bash
cd example
mix deps.get
mix firmware
```

## Deploying to Board

```bash
# Upload firmware to board (replace with your board's IP)
mix upload --board <board-ip>

# Or if using SSH alias
mix upload --board nerves.local
```

## Testing the NIF

Once deployed, you can test the NIF by connecting to the board:

```bash
# SSH into the board
ssh nerves.local

# Or run IEx remotely
iex --remsh nerves.local --cookie nerves

# Test the NIF
iex> Sscmex.hello()
{:ok, "Hello from SSCMEx NIF!"}
```

You should see logs showing:
```
Starting Sscmex Example Application...
SSCMEx NIF loaded successfully!
NIF test passed: Hello from SSCMEx NIF!
```

## What This Tests

1. **NIF Loading**: Verifies that the NIF shared library can be loaded on the SG2002
2. **NIF Function Call**: Calls the `hello/0` NIF function and verifies it returns the expected result
3. **Logging**: All operations are logged to the console for debugging

## Next Steps

Once this example works on real hardware:

1. Add TPU SDK initialization code
2. Add camera frame capture from V4L2
3. Add TPU inference calls for object detection
4. Add Nx tensor conversion for binary data handling
5. Add camera preview/testing functions

## Troubleshooting

### NIF fails to load

Check that the NIF was compiled for the correct architecture (RISCV):

```bash
file _build/dev/lib/sscmex/priv/sscmex_nif.so
# Should show: ELF 64-bit LSB shared object, UCB RISC-V, ...
```

### Connection issues

```bash
# Check board is accessible
ping nerves.local

# Verify SSH keys
ssh nerves.local
```

### Logs

```bash
# View full logs on the board
ssh nerves.local "journalctl -f -u sscmex_example.service"

# Or view ring buffer
dmesg | grep -i sscmex
```
