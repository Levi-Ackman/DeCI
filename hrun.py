import os
import argparse

def execute_sh_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".sh"):
                sh_file_path = os.path.join(root, file)
                print(f"Executing {sh_file_path}")
                os.system(f"chmod +x {sh_file_path}")  # Ensure the script is executable
                os.system(sh_file_path)
def main():
    parser = argparse.ArgumentParser(description="Execute .sh files in a specified directory")
    parser.add_argument("--opt", default=1,type=int, required=False, help="Specify the option for training")
    args = parser.parse_args()
    if args.opt == 1:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/Attn/PSCRAttn/Abide"
    elif args.opt == 2:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/Attn/PSCRAttn/other"
    elif args.opt == 3:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/Attn/STAGIN/Abide"
    elif args.opt == 4:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/Attn/STAGIN/other"
    elif args.opt == 5:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/dFC/BrainTGL/Abide"
    elif args.opt == 6:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/dFC/BrainTGL/other"
    elif args.opt == 7:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/dFC/MDGL/Abide"
    elif args.opt == 8:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/dFC/MDGL/other"
    elif args.opt == 9:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/Multi_View/MVHO/Abide"
    elif args.opt == 10:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/Multi_View/MVHO/other"
    elif args.opt == 11:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/Multi_View/SimMVF/Abide"
    elif args.opt == 12:
        scripts_directory = "/data/gqyu/FMRI/TMI_Re/base/scripts/Multi_View/SimMVF/other"    
    else:
        print("Invalid option specified.")
        return
    execute_sh_files_in_directory(scripts_directory)

if __name__ == "__main__":
    main()
