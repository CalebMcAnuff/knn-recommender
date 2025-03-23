# Necessary files to run the fastapi 
- the csv
- knn.pkl
- requirements.txt
- mainv3.py

# To Run on AWS
sudo apt update && upgrade
sudo apt install python3.12-venv
sudo apt install python3.12-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nohup uvicorn mainv3:app --host 0.0.0.0 --port 5000 > output.log 2>&1 & 
## Then open security group

