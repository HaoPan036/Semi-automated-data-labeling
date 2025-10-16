# 🔧 GitHub推送错误解决方案

## ❌ 错误信息
```
fatal: unable to access 'https://github.com/HaoPan036/Semi-automated-data-labeling.git/': 
LibreSSL SSL_connect: SSL_ERROR_SYSCALL in connection to github.com:443
```

## 🛠️ 解决方案 (按优先级排序)

### 方案1: 使用SSH连接 (推荐) 🔑

这是最安全稳定的方法：

```bash
# 1. 生成SSH密钥 (如果还没有)
ssh-keygen -t ed25519 -C "your.email@example.com"
# 按Enter键使用默认路径，可以设置密码或直接按Enter

# 2. 将SSH公钥添加到SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 3. 复制SSH公钥
cat ~/.ssh/id_ed25519.pub
# 复制输出的整个公钥内容

# 4. 在GitHub中添加SSH密钥
# - 登录GitHub → Settings → SSH and GPG keys → New SSH key
# - 粘贴刚复制的公钥内容

# 5. 更改远程仓库URL为SSH
cd /Users/hao/Desktop/code/Semi-automated/semi_auto_label_demo
git remote set-url origin git@github.com:HaoPan036/Semi-automated-data-labeling.git

# 6. 测试SSH连接
ssh -T git@github.com

# 7. 推送代码
git push -u origin main
```

### 方案2: 修复HTTPS连接 🌐

```bash
cd /Users/hao/Desktop/code/Semi-automated/semi_auto_label_demo

# 尝试更新Git配置
git config --global http.version HTTP/1.1
git config --global http.postBuffer 157286400
git config --global http.sslVerify true

# 清除Git凭据缓存
git config --global --unset credential.helper
git config --global credential.helper osxkeychain

# 重新尝试推送
git push -u origin main
```

### 方案3: 使用GitHub CLI (简单) 📱

```bash
# 1. 安装GitHub CLI (如果没安装)
brew install gh

# 2. 登录GitHub
gh auth login
# 选择GitHub.com → HTTPS → 使用浏览器登录

# 3. 推送代码
gh repo create Semi-automated-data-labeling --public --source=. --remote=origin --push
```

### 方案4: 临时网络修复 🔧

```bash
cd /Users/hao/Desktop/code/Semi-automated/semi_auto_label_demo

# 临时禁用SSL验证 (不推荐用于生产)
git config --global http.sslVerify false
git push -u origin main

# 推送成功后恢复SSL验证
git config --global http.sslVerify true
```

## 🏆 最佳实践推荐

### 使用SSH (方案1) 的优势：
- ✅ 更安全可靠
- ✅ 避免SSL问题
- ✅ 不需要每次输入密码
- ✅ GitHub官方推荐

### 快速SSH设置指南：
1. **生成密钥**: `ssh-keygen -t ed25519 -C "your.email@example.com"`
2. **复制公钥**: `cat ~/.ssh/id_ed25519.pub`
3. **添加到GitHub**: Settings → SSH keys → 粘贴公钥
4. **更换URL**: `git remote set-url origin git@github.com:HaoPan036/Semi-automated-data-labeling.git`
5. **推送**: `git push -u origin main`

## 🔍 故障排除

如果问题仍然存在，可能的原因：
- 🌐 网络防火墙或代理设置
- 🔒 公司网络限制
- ⏰ 临时的GitHub服务问题
- 🖥️ 本地Git配置问题

## 📞 需要帮助？

如果以上方案都不行，请尝试：
1. 重启终端和网络连接
2. 检查是否有VPN或代理影响
3. 确认GitHub仓库确实存在且有推送权限
4. 联系网络管理员检查防火墙设置