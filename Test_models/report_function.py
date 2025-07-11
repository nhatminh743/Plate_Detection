def compare_txt_files(file1_path, file2_path, long_report = True):
    """
    Compare two text files line by line and report matching accuracy.

    Parameters:
        file1_path (str): Path to the first TXT file.
        file2_path (str): Path to the second TXT file.
    """
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # Make sure both files have the same number of lines
    total_lines = min(len(lines1), len(lines2))
    if len(lines1) != len(lines2):
        print(f"⚠️ Warning: Files have different number of lines (file1: {len(lines1)}, file2: {len(lines2)})")
        print(f"→ Comparing only first {total_lines} lines.\n")

    match_count = 0
    mismatch_count = 0

    for idx in range(total_lines):
        line1 = lines1[idx].strip()
        line2 = lines2[idx].strip()
        if line1 == line2:
            match_count += 1
        else:
            mismatch_count += 1
            if long_report:
                print(f"Mismatch at line {idx + 1}:")
                print(f"  File 1: {line1}")
                print(f"  File 2: {line2}")
                print("-" * 40)

    accuracy = (match_count / total_lines) * 100 if total_lines > 0 else 0

    print("\n=== Comparison Report ===")
    print(f"Total Lines Compared : {total_lines}")
    print(f"Matching Lines       : {match_count}")
    print(f"Mismatched Lines     : {mismatch_count}")
    print(f"Accuracy             : {accuracy:.2f}%")
