# Import all config files
import Config

# If running under Nerves, use the target config
if Mix.target() != :host do
  import_config "target.exs"
else
  import_config "host.exs"
end

# Import secrets (WiFi credentials, etc.)
import_config "secrets.exs"
