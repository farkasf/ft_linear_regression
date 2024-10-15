#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

animation() {
    local pid=$!
    local delay=0.3
    local spinstr='|/-\'
    while true; do
        for i in $(seq 0 3); do
            echo -ne "${YELLOW}[ ${spinstr:i:1} ] ${1}${NC}\033[K\r"
            sleep $delay
        done
    done
}

start_animation() {
    animation "$1" &
    ANIMATION_PID=$!
}

stop_animation() {
    kill $ANIMATION_PID
    wait $ANIMATION_PID 2>/dev/null
    echo -ne "\r\033[K"
    echo -e "${GREEN}$1${NC}"
}

start_animation "creating workdir folder"
mkdir -p workdir
sleep 1
stop_animation "workdir folder created"

cd workdir

start_animation "copying files"
cp -r ../sources/* .
cp -r ../resources/* .
sleep 1
stop_animation "files copied"

start_animation "creating virtual environment"
python3 -m venv v_env
sleep 1
stop_animation "virtual environment created"

source v_env/bin/activate

if [ -d "v_env/bin" ] && [ -f "v_env/pyvenv.cfg" ]; then
    echo -e "${GREEN}virtual environment successfully activated${NC}"
else
    echo -e "${RED}virtual environment not activated${NC}"
    exit 1
fi

start_animation "installing required packages"
{
    pip install --upgrade pip --quiet > /dev/null 2>&1
    pip install -r requirements.txt --quiet > /dev/null 2>&1
}
stop_animation "required packages installed"

echo -e "${GREEN}setup completed.${NC}"
echo -e "\n    ${YELLOW}to change to the workdir directory, run:${NC} cd workdir\n"
