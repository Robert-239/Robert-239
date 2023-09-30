echo "Hello World"

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

winget install neovim
winget install vscodium
winget install wsl
winget install nodejs

choco install make 
echo "git setup"
winget install Git.Git

git config --global user.name "Robert Wilson"
git config --global user.email "robertwilson02@icloud.com"

git config --global user.name
git config --global user.email

echo "wsl distro download"
wsl --install -d Ubuntu
