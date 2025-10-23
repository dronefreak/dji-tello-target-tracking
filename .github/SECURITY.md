# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them responsibly using one of the following methods:

### Preferred Method: Private Security Advisory

1. Go to the [Security tab](https://github.com/dronefreak/dji-tello-target-tracking/security) of this repository
2. Click "Report a vulnerability"
3. Fill out the form with details about the vulnerability
4. Submit the report

### Alternative Method: Email

If you prefer email, send your report to the project maintainers through GitHub's contact system:

- Open a private discussion with maintainers
- Include "[SECURITY]" in the subject line
- Provide detailed information about the vulnerability

### What to Include

Please include the following information in your report:

- **Type of vulnerability** (e.g., code injection, privilege escalation, drone control hijacking)
- **Affected component** (e.g., detector, tracker, drone_controller)
- **Steps to reproduce** the issue
- **Potential impact** (e.g., unauthorized drone control, data exposure)
- **Suggested fix** (if you have one)
- **Your contact information** for follow-up questions

### Example Report

```markdown
**Vulnerability Type:** Command Injection

**Affected Component:** drone_controller.py, line 234

**Description:**
The drone controller accepts user input for custom commands without
proper sanitization, potentially allowing arbitrary command execution.

**Steps to Reproduce:**

1. Initialize DroneController
2. Call custom_command() with malicious input: "; rm -rf /"
3. Command is executed without validation

**Impact:**
High - Could allow arbitrary code execution on the host system

**Suggested Fix:**
Add input validation and use parameterized commands instead of
string concatenation.

**Contact:** [your-email@example.com]
```

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity (see below)

### Severity Levels

| Severity     | Response Time | Examples                                          |
| ------------ | ------------- | ------------------------------------------------- |
| **Critical** | 1-3 days      | Remote code execution, unauthorized drone control |
| **High**     | 1-2 weeks     | Authentication bypass, data exposure              |
| **Medium**   | 2-4 weeks     | Denial of service, information disclosure         |
| **Low**      | 4-8 weeks     | Minor information leaks, edge cases               |

## Security Considerations for This Project

### Drone Control Security

This project involves controlling physical drones, which introduces unique security concerns:

**Physical Safety Risks:**

- Unauthorized drone control could cause physical harm
- Malicious commands could damage the drone or nearby property
- GPS spoofing or signal interference could lead to crashes

**Mitigation Measures:**

- All flight commands include safety checks (battery, altitude limits)
- Emergency stop functionality (ESC key)
- Configurable flight boundaries
- Manual override always available

**Known Limitations:**

- No authentication for drone connection (Tello limitation)
- Video stream is unencrypted (Tello limitation)
- Local network communication only

### Network Security

**WiFi Considerations:**

- Tello drones create their own WiFi network
- Connection is not encrypted by default
- Anyone in range can potentially connect

**Recommendations:**

- Fly in controlled environments
- Use unique SSID and password if your drone model supports it
- Monitor for unexpected connections
- Avoid flying in crowded areas with many WiFi networks

### Data Privacy

**What Data is Collected:**

- ❌ No telemetry is sent to external servers
- ❌ No personal information is collected
- ✅ All processing happens locally
- ✅ Video/images are stored locally only

**User Responsibility:**

- Ensure compliance with local privacy laws when recording
- Obtain consent before recording people
- Handle recorded data responsibly

### Software Supply Chain

**Dependencies:**

- We use well-maintained, popular libraries (PyTorch, OpenCV, ultralytics)
- Regular dependency updates for security patches
- Automated vulnerability scanning via GitHub Dependabot

**Recommendations:**

- Regularly update dependencies: `pip install --upgrade -r requirements.txt`
- Review dependency changes before updating
- Use virtual environments to isolate dependencies

## Secure Development Practices

### Code Review

- All code changes require review before merging
- Security-sensitive changes require additional scrutiny
- Automated tests must pass before merge

### Testing

- Unit tests for core functionality
- Integration tests for end-to-end workflows
- Manual testing with real hardware when applicable

### Dependencies

- Pin dependency versions in requirements.txt
- Review dependency updates for security advisories
- Use `pip-audit` to check for known vulnerabilities

```bash
# Check for vulnerabilities in dependencies
pip install pip-audit
pip-audit
```

## Vulnerability Disclosure Policy

### Our Commitment

- We will acknowledge receipt of your report within 48 hours
- We will keep you informed about the progress of fixing the issue
- We will credit you in the security advisory (unless you prefer to remain anonymous)
- We will not take legal action against security researchers who:
  - Report vulnerabilities responsibly
  - Do not exploit vulnerabilities beyond proof of concept
  - Do not access, modify, or delete data without permission

### Public Disclosure

- We will work with you to determine an appropriate disclosure timeline
- Typical timeline: Fix is developed → Patch is released → Public disclosure
- We aim to release fixes within 90 days of initial report
- Critical vulnerabilities may be disclosed sooner with a workaround

### Recognition

Security researchers who responsibly disclose vulnerabilities will be:

- Credited in the security advisory (with permission)
- Listed in SECURITY_HALL_OF_FAME.md (if applicable)
- Acknowledged in release notes

## Security Best Practices for Users

### Installation

```bash
# Always install from official sources
pip install -r requirements.txt

# Verify package integrity
pip check

# Use virtual environments
python -m venv venv
source venv/bin/activate
```

### Configuration

```python
# Use configuration files instead of hardcoding
config = Config()
config.load_from_file("config.yaml")

# Set appropriate safety limits
config.max_speed = 50  # Don't exceed safe speeds
config.min_battery = 20  # Land before battery is too low
```

### Flying Safely

```python
# Always use try-except for drone operations
try:
    drone.connect()
    drone.takeoff()
    # ... flight operations
except Exception as e:
    print(f"Error: {e}")
    drone.emergency_stop()
finally:
    drone.land()
    drone.disconnect()
```

### Updating

```bash
# Keep software updated for security patches
git pull origin main
pip install --upgrade -r requirements.txt

# Check for security advisories
# Visit: https://github.com/dronefreak/dji-tello-target-tracking/security/advisories
```

## Known Security Limitations

### Hardware Limitations (Tello Drone)

These limitations are inherent to the DJI Tello platform:

1. **No Authentication**: Anyone can connect to the Tello's WiFi
2. **Unencrypted Video**: Video stream is sent over unencrypted UDP
3. **Limited Range**: ~100m range (security through obscurity)
4. **No GPS**: Cannot implement geofencing in this codebase

### Software Limitations

1. **Object Detection**: May misidentify objects, leading to tracking errors
2. **Network Latency**: Video delay can cause control lag
3. **Single Client**: Only one client can control the drone at a time (race conditions possible)

## Security Updates

Subscribe to security advisories:

- Watch this repository on GitHub
- Enable "Security alerts" in your notification settings
- Check the [Security tab](https://github.com/dronefreak/dji-tello-target-tracking/security) regularly

## Additional Resources

- [OWASP IoT Security Project](https://owasp.org/www-project-internet-of-things/)
- [Drone Security Best Practices](https://www.cisa.gov/uas-cybersecurity)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security.html)

## Questions?

If you have questions about this security policy, please open a public issue (for general questions) or contact maintainers privately (for security concerns).

---

**Last Updated**: October 2025
