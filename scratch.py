import json

# Example JSON result (can also be loaded from a file or API response)
data = {
    "results": {
        "0010_00004_b": {
            "text": "59-F1 07509"
        }
    }
}

# Extract the filename and text
filename = list(data["results"].keys())[0]
plate_text = data["results"][filename]["text"]

print(f"Filename: {filename}")
print(f"Extracted Text: {plate_text}")