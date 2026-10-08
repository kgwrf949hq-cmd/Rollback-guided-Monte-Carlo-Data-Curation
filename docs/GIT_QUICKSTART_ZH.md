# 本项目的 Git 与 GitHub 入门

Git 是本地版本管理工具；GitHub 是保存远程仓库、查看差异和协作的网站。此次双模型版本位于 `dual-model/paired-pilot-validated-20261008` 分支，代码和结果入口是 `experiments/dual_model_paired/README.md`。

## 先记住这几个词

| 名称 | 作用 |
|---|---|
| 仓库 repository | 项目文件及其修改历史 |
| 分支 branch | 一条可以独立推进的版本线；`main` 通常作为主线 |
| 暂存 add | 选出准备放进下一次提交的修改 |
| 提交 commit | 在本地保存一次带说明的版本记录 |
| 推送 push | 把本地提交上传到 GitHub |
| 拉取 pull | 获取并整合远程更新到当前分支 |
| PR / Pull Request | 请求把一个分支的修改合并到另一个分支，方便先审阅 |

保存文件后还需要 `add → commit → push`，网站才能看到修改。推送分支不等于合并进 `main`。

## 终端：获取此次双模型版本

如果已经有这个仓库，在仓库文件夹打开终端；没有则先克隆：

```bash
git clone https://github.com/kgwrf949hq-cmd/Rollback-guided-Monte-Carlo-Data-Curation.git
cd Rollback-guided-Monte-Carlo-Data-Curation
```

然后获取远程分支并切换：

```bash
git status
git fetch origin
git switch --track origin/dual-model/paired-pilot-validated-20261008
```

如果本地已存在这个分支，最后一条改成：

```bash
git switch dual-model/paired-pilot-validated-20261008
```

`origin` 是这个仓库的远程地址别名。`fetch` 更新远程分支信息，不会直接修改当前工作文件。切换前若 `git status` 显示有尚未提交的修改，先保存成合适的提交，避免把两项工作混在一起。

## 终端：继续完善双模型策略

在上述双模型分支上，可以继续提交，也可以新建下一阶段分支：

```bash
git switch -c dual-model/next-comparison
```

编辑文件后，例如修改策略说明：

```bash
git status
git diff
git add experiments/dual_model_paired/README.md
git diff --cached
git commit -m "Clarify the dual-model collection protocol"
git push -u origin dual-model/next-comparison
```

`add` 后写实际改动的文件路径；多文件可以列多个路径。`git diff` 查看未暂存内容，`git diff --cached` 查看这次将提交的内容。首次 `push -u` 建立本地和远程分支的对应关系，后续同一分支通常只需 `git push`。

## 终端：同步与查看历史

```bash
git status                  # 当前分支、修改和同步状态
git branch -a               # 本地与已获取的远程分支
git log --oneline -5         # 最近五次提交
git remote -v               # 远程仓库地址
git pull --ff-only          # 当前分支可以直接前进时才拉取
```

若 `pull --ff-only` 提示分叉或冲突，先查看双方提交再决定 merge 或 rebase。日常工作不需要强制推送。

PR 在网页合并后，工作区干净时同步主分支：

```bash
git switch main
git pull --ff-only origin main
```

如果 Git 要求设置身份，按本人信息配置一次 `git config --global user.name "你的名字"` 和 `git config --global user.email "你的提交邮箱"`。这是提交署名；GitHub 登录认证是另一件事，推送需要已配置的凭据、SSH 或令牌，不能把 GitHub 账户密码当作 HTTPS Git 密码。

## 网页：看代码、修改文件、发起合并

1. 打开项目，在 `Code` 页左上分支下拉框选择 `dual-model/paired-pilot-validated-20261008`，进入 `experiments/dual_model_paired/` 阅读说明和结果。
2. 查看历史：打开提交记录查看每次修改；查看文件差异时，绿色通常为新增，红色为删除。
3. 修改小型说明或配置：打开文件点铅笔图标，编辑后点 `Commit changes`。需要独立工作时，选择创建新分支；也可先在分支下拉框输入新名称创建分支。多文件代码修改和运行测试更适合终端/编辑器。
4. 提出合并：进入 `Pull requests → New pull request`，选择 `base: main`、`compare: 你的分支`，检查差异，填写标题与说明。尚待讨论可创建 Draft PR。
5. 审阅确认后再点击 `Merge pull request`。创建 PR 本身不会合并；你和师兄可以先查看、讨论和继续提交。

网页上提交的改动不会自动进入服务器或本地文件夹；需在相应仓库、相应分支执行 `git pull --ff-only`。本地 `commit` 也不会自动出现在网页，需执行 `git push`。

官方参考：[Git switch](https://git-scm.com/docs/git-switch)、[Git push](https://git-scm.com/docs/git-push)、[网页管理分支](https://docs.github.com/en/pull-requests/how-tos/commit-changes/managing-branches-within-your-repository)、[创建 PR](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request)。
