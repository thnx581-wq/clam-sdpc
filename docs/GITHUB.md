# 发布到自己的 GitHub

此目录应作为一个独立仓库发布。原 CLAM 仓库的 `origin` 指向 Mahmood Lab；本精简版不会向其推送。原始工程保留在本机，发布目录只含源码、文档、配置和测试。

需要确定：你的 GitHub 用户名、目标仓库名、公开或私有。如果目标仓库已存在，提供其 URL。身份验证在你自己的浏览器或 Git Credential Manager 中完成，**不要在聊天中提供 token、密码或 SSH 私钥**。

若安装了 GitHub CLI：

```powershell
gh auth login --hostname github.com --git-protocol https --web
gh repo create YOUR_ACCOUNT/clam-sdpc --private --source . --remote origin --push
```

将 `--private` 换成 `--public` 可公开。上面创建仓库的命令仅在你已选好名称、可见性且完成登录后执行。

若没有 GitHub CLI，在 GitHub 网页新建空仓库，不预置 README/License/.gitignore；然后在本目录执行：

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/clam-sdpc.git
git push -u origin main
```

Git for Windows 会按本机配置调用凭据管理器登录。若此目录尚无提交，先执行：

```powershell
git init -b main
git add .
git commit -m "Extract portable CLAM SDPC preprocessing workflow"
```

提交身份由你的 Git 设置决定。若缺失，使用自己的姓名和 GitHub 已验证邮箱（或 GitHub noreply 邮箱）设置本仓库的 `user.name/user.email`；不要替其他作者杜撰身份。

提交前检查 `git status --short` 和 `git ls-files`。`.gitignore` 已排除常见权重、切片、结果、DLL 和表格；实际患者清单、密钥与自定义私有路径同样不应上传。保留 LICENSE.md 和 NOTICE.md。只有模型代码随仓库发布，模型权重依照各模型的使用条款单独获取。
