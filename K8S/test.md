# 109 机器 K8S 安装后运行测试报告

## 测试结论

109 机器当前可以正常登录和访问，系统资源、K8S 基础服务、CRI-O、K8S API 健康检查、节点 kubelet 健康检查、MUSA 驱动层和 PyTorch MUSA 最小算子测试均通过。

需要注意的是，当前用户 `o_mabin` 没有默认 `~/.kube/config`，直接执行 `kubectl get nodes` 会失败；使用 `/etc/kubernetes/kubeconfig` 可以访问 API Server，`/readyz`、`/livez` 均返回正常，但该 kubeconfig 身份权限不足，不能列出 nodes/pods，也不能创建测试 Pod。若后续需要执行 `kubectl run`、`kubectl create deployment`、`kubectl logs` 等完整工作负载测试，需要管理员提供 kubeconfig 或补充 RBAC 授权。

## 基本信息

| 项目 | 结果 |
| --- | --- |
| 测试时间 | 2026-04-30 00:31-00:46 UTC |
| 测试机器 | ICT109 |
| 测试用户 | o_mabin |
| 系统内核 | Linux 5.15.0-164-generic |
| CPU | Intel Xeon Gold 6430，128 CPUs |
| 内存 | 503 GiB，总可用约 483 GiB |
| `/home` 空间 | 19T，总使用 368G，剩余 17T |
| Python | Python 3.10.12 |

## 测试项结果

| 序号 | 测试项 | 检查内容 | 结果 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 登录检查 | 当前会话是否在 109 机器上 | 通过 | `hostname` 为 `ICT109` |
| 2 | 基础资源 | 内存、磁盘、系统负载 | 通过 | 资源充足，`/home` 使用率约 3% |
| 3 | K8S kubelet | `systemctl is-active kubelet` | 通过 | `active`，自 2026-04-29 07:46 UTC 起运行 |
| 4 | K8S CRI-O | `systemctl is-active crio` | 通过 | `active`，自 2026-04-29 11:26 UTC 起运行 |
| 5 | K8S API readyz | `kubectl --kubeconfig /etc/kubernetes/kubeconfig get --raw=/readyz?verbose` | 通过 | etcd、informer、RBAC bootstrap、APIServices 等检查均为 `ok` |
| 6 | K8S API livez | `kubectl --kubeconfig /etc/kubernetes/kubeconfig get --raw=/livez` | 通过 | 返回 `ok` |
| 7 | K8S 版本 | `kubectl --kubeconfig /etc/kubernetes/kubeconfig get --raw=/version` | 通过 | `v1.29.8-53+1436ff63b04a98-dirty` |
| 8 | kubelet healthz | `curl -s http://127.0.0.1:10248/healthz` | 通过 | 返回 `ok` |
| 9 | Pod 落地情况 | 查看 `/var/log/pods` | 通过 | 发现 calico、dns、kube-proxy、monitoring、logging、nginx 等 Pod 日志目录 |
| 10 | 创建测试 Pod 权限 | `kubectl auth can-i create pods -A` | 未通过 | 返回 `no`，当前身份无创建 Pod 权限 |
| 11 | 读取 Pod 权限 | `kubectl auth can-i get pods -A` | 未通过 | 返回 `no`，当前身份无读取 Pod 权限 |
| 12 | CRI-O 运行时 CLI | `crictl info` / `crictl ps -a` | 未通过 | 普通用户无 `/var/run/crio/crio.sock` 权限；sudo 需要密码 |
| 13 | MUSA 驱动 | `mthreads-gmi` | 通过 | 识别 2 张 MTT S3000 |
| 14 | MUSA Runtime | `/usr/local/musa/bin/musa_runtime_version` | 通过 | Runtime 版本 4.2.0 |
| 15 | `musaInfo` | 加 `LD_LIBRARY_PATH=/usr/local/musa/lib` 后执行 | 通过 | 识别 2 张 MTT S3000 |
| 16 | Python 依赖 | `torch`、`torch_musa`、`transformers`、`accelerate` | 通过 | 项目环境脚本下均可用 |
| 17 | PyTorch MUSA | 最小张量创建与求和 | 通过 | `musa_count=2`，`tensor_device=musa:0`，`tensor_sum=4.0` |
| 18 | 原项目 preflight | `preflight_check.py` | 通过 | 单卡/双卡可见性均为 true |

## 关键命令输出摘要

### K8S

```text
systemctl is-active kubelet
active

systemctl is-active crio
active

kubectl --kubeconfig /etc/kubernetes/kubeconfig get --raw=/livez
ok

kubectl --kubeconfig /etc/kubernetes/kubeconfig get --raw=/readyz
ok

curl -s http://127.0.0.1:10248/healthz
ok
```

K8S API 详细健康检查：

```text
kubectl --kubeconfig /etc/kubernetes/kubeconfig get --raw=/readyz?verbose
[+]ping ok
[+]log ok
[+]etcd ok
[+]etcd-readiness ok
[+]api-ccos-apiserver-available ok
[+]api-ccos-oauth-apiserver-available ok
[+]informer-sync ok
[+]poststarthook/rbac/bootstrap-roles ok
[+]poststarthook/apiservice-status-available-controller ok
[+]shutdown ok
readyz check passed
```

K8S 版本：

```text
kubectl --kubeconfig /etc/kubernetes/kubeconfig get --raw=/version
gitVersion: v1.29.8-53+1436ff63b04a98-dirty
platform: linux/amd64
```

当前用户默认没有 kubeconfig：

```text
kubectl config view
clusters: null
contexts: null
current-context: ""
users: null
```

使用系统 kubeconfig 查询资源时，API 可达但权限不足：

```text
Error from server (Forbidden): nodes is forbidden:
User "system:serviceaccount:ccos-machine-config-operator:node-bootstrapper"
cannot list resource "nodes" in API group "" at the cluster scope
```

当前 kubeconfig 也没有创建测试 Pod/Deployment 的权限：

```text
kubectl --kubeconfig /etc/kubernetes/kubeconfig auth can-i create pods -A
no

kubectl --kubeconfig /etc/kubernetes/kubeconfig auth can-i create deployments -A
no
```

本节点已有 K8S Pod 日志目录，说明 kubelet 已经在本机落地运行工作负载：

```text
/var/log/pods/ccos-calico_calico-node-qtcxx_...
/var/log/pods/ccos-dns_dns-default-g847n_...
/var/log/pods/ccos-kube-proxy_ccos-kube-proxy-8zmm2_...
/var/log/pods/ccos-monitoring_node-exporter-4lld5_...
/var/log/pods/ccos-logging_default-logging-fluentbit-g4cgn_...
/var/log/pods/default_nginx-deploy-7cc497769-rkqnq_...
```

nginx 测试 Pod 的日志文件存在，但普通用户不能读取日志内容：

```text
/var/log/pods/default_nginx-deploy-7cc497769-rkqnq_.../nginx/0.log
-rw------- 1 nobody nogroup ...
tail: Permission denied
```

### MUSA 设备

```text
mthreads-gmi
Driver Version: 3.1.0-rc4.2.0-server-Ubuntu
0  MTT S3000  32768MiB
1  MTT S3000  32768MiB
```

```text
/usr/local/musa/bin/musa_runtime_version
version: 4.2.0
```

```text
LD_LIBRARY_PATH=/usr/local/musa/lib /usr/local/musa/bin/musaInfo
device#0 MTT S3000
device#1 MTT S3000
```

### PyTorch MUSA

测试命令使用项目已有环境脚本：

```text
moerxiancheng-clj-xyj-proj/clj-proj/train-infer-estimation-release-2026-04-11/tools/python_with_env.sh -c "import torch; import torch_musa; ..."
```

输出摘要：

```text
torch 2.5.0
musa_count 2
tensor_device musa:0
tensor_sum 4.0
```

## 注意事项

1. 直接执行 `/usr/local/musa/bin/musaInfo` 会因为缺少 `libmusart.so.4` 加载失败，需要先设置 `LD_LIBRARY_PATH=/usr/local/musa/lib`，或使用项目里的 `tools/python_with_env.sh`。
2. 直接用系统 `python3 import torch` 会因为缺少 MUSA 相关动态库路径失败；使用项目环境脚本后可以正常导入并使用 MUSA。
3. `matplotlib` 当前未安装；如果后续需要重新生成图表，可能需要补装或进入对应项目环境。
4. `nvidia-smi` 无法连接 NVIDIA driver；该机器主要使用摩尔线程 MUSA 设备，此项不作为异常判断。
5. 当前用户无法直接列出 K8S nodes/pods，也无法创建测试 Pod/Deployment；建议由管理员补充只读和测试命名空间权限后，再执行完整 `kubectl run nginx/busybox` 工作负载测试。
6. `crictl` 已安装，但普通用户不能访问 `/var/run/crio/crio.sock`；如果需要查看本机容器运行时详情，需要 sudo 密码或加入对应权限组。

## 测试产物

- Host 真实环境 preflight：`moerxiancheng-clj-xyj-proj/K8S/preflight_109_20260430_host.json`
- 沙箱内 preflight：`moerxiancheng-clj-xyj-proj/K8S/preflight_109_20260430.json`，仅用于对照，不作为正式结论

## 给群里的简短反馈

```text
我已在 109/ICT109 上做了一轮基础验证：机器可正常访问，kubelet 和 crio 均为 active，K8S API livez/readyz 返回 ok，readyz 详细检查中 etcd、informer、RBAC bootstrap、APIServices 等均为 ok；本机 /var/log/pods 下已有 calico、dns、kube-proxy、monitoring、logging、default/nginx-deploy 等 Pod 目录，说明 K8S 工作负载已落地运行。MUSA 驱动可识别 2 张 MTT S3000，项目环境脚本下 torch/torch_musa 可正常导入，PyTorch MUSA 最小张量测试通过，能看到 2 张卡。

目前注意到两点：当前用户没有默认 kubeconfig，系统 kubeconfig 能访问 API 但 RBAC 不允许列 nodes/pods，也不能创建测试 Pod/Deployment；另外 MUSA/PyTorch 程序需要使用项目里的 tools/python_with_env.sh 或设置 LD_LIBRARY_PATH，否则直接 python3/import torch 可能因动态库路径缺失失败。
```

## 原始通知

```text
@蓝色极光 麻烦杜老师明天也检查一下吧
也请各个课题使用109这台机器的人都试一下吧，看看之前的程序啥的是否还能正常使用之类的，以防万一
@明远 @0x01 请科大和电信的老师和同学也试一下109吧
摩尔线程的那台机器我们昨天装了K8S，计算所的老师让我们多方确认一下那台机器现在是否正常运行
```
