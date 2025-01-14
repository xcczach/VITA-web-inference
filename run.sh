> process_log.txt
port=${1:-"9234"}
api_name=vita
hangup_timeout_sec=900
hangup_interval_sec=60
rm -f nohup.out
nohup python main.py --port $port --api-name $api_name --hangup-timeout-sec $hangup_timeout_sec --hangup-interval-sec $hangup_interval_sec &
echo "API Server to be started on port $port. Post to http://localhost:$port/$api_name to use the API."
echo "Server PID: $!" >> process_log.txt
echo "Server PID: $!, saved to process_log.txt"