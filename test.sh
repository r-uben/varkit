#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running R script...${NC}"
if command -v Rscript &> /dev/null; then
    Rscript test.R
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}R script completed successfully${NC}"
    else
        echo -e "${RED}Error running R script${NC}"
        exit 1
    fi
else
    echo -e "${RED}Rscript not found. Please install R or add it to your PATH${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Running MATLAB script...${NC}"
if command -v matlab &> /dev/null; then
    matlab -nodisplay -nosplash -nodesktop -r "run('test.m'); exit;" | cat
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}MATLAB script completed successfully${NC}"
    else
        echo -e "${RED}Error running MATLAB script${NC}"
        exit 1
    fi
else
    echo -e "${RED}MATLAB not found. Please install MATLAB or add it to your PATH${NC}"
    exit 1
fi

echo -e "\n${GREEN}All scripts completed successfully!${NC}" 