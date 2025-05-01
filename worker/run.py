import importlib
import os
import sys

def get_curr_working_dir():
    return os.getcwd()

def run():
    cwd = get_curr_working_dir()
    sys.path.append(cwd)
    sys.path.append(f"{cwd}/challenge_data/challenge_1/evaluation_script")

    challenge_id = 1
    phase_codename = "test"
    annotation_file_path = f"{cwd}/annotations/test_phase_annotations.zip"
    user_submission_file_path = f"{cwd}/annotations/test_phase_submission.zip"

    CHALLENGE_IMPORT_STRING = "challenge_data.challenge_1.evaluation_script"
    challenge_module = importlib.import_module(CHALLENGE_IMPORT_STRING)

    EVALUATION_SCRIPTS = {challenge_id: challenge_module}

    print("🚀 正在运行本地评估脚本 evaluate(...)")
    result = EVALUATION_SCRIPTS[challenge_id].evaluate(
        annotation_file_path,
        user_submission_file_path,
        phase_codename,
        submission_metadata={}
    )
    print("✅ 本地评估完成，结果如下：")
    print(result)

if __name__ == "__main__":
    run()
