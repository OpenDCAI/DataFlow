import json

def detect_format(filepath):
    """Attempt to detect whether the file is in standard JSON format or ND-JSON format."""
    with open(filepath, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
        try:
            # Try to parse the first line as a JSON object
            json.loads(first_line)
            # If no exception is raised, it's likely ND-JSON format
            return 'ndjson'
        except json.JSONDecodeError:
            # If the first line cannot be parsed as a JSON object, it might be standard JSON format
            file.seek(0)  # Reset file pointer to the beginning
            first_bytes = file.read(1024)  # Read the first 1024 bytes
            if first_bytes.startswith('[') and first_bytes.endswith(']'):
                return 'standard'
            else:
                raise ValueError("The file is neither in standard JSON nor ND-JSON format")
            

def read_json_file(filepath):
    format_type = detect_format(filepath)
    
    if format_type == 'ndjson':
        with open(filepath, 'r', encoding='utf-8') as file:
            return [json.loads(line.strip()) for line in file]
    elif format_type == 'standard':
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)