# tar.gz 파일들의 파일명과 같은 폴더 아래에 압축해제 하는 스크립트
for file in *.tar.gz; do mkdir "${file%%.*}"; tar -xvzf "${file}" -C "${file%%.*}" ; done