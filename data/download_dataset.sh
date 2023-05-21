python -c 'from mega import Mega; Mega().download_url("https://mega.nz/file/kFsjlJbR#UAXUK2A_rNHf-v5ewD4MR9czJBZ9fDUv9ytWpmRSeOw")'
unzip coad_dataset.zip
mv coad_dataset/* .
rm -rf __MACOSX
rm -rf coad_dataset
rm -rf .megadown
rm coad_dataset.zip