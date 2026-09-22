
## Directory Structure Script

### cmd to get directory structure
```bash
tree -I "$(paste -sd '|' ignore.txt)"
```