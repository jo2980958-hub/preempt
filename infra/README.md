# infra

Shell scripts that reconcile AWS to a desired state, print what they did, and are safe
to re-run. The region comes from `AWS_REGION`, the account from `EXPECTED_ACCOUNT`, and
everything is tagged `Project=opencv26` and `Product=preempt`. Set `DRY_RUN=1` to see the
commands without running them.

| Script | What it does |
|---|---|
| `ecr.sh <product>` | Create or reuse a private ECR repo, log in, buildx, push |
| `apprunner.sh <product>` | Create or update the App Runner service from ECR, wait, print the URL |
| `common.sh` | Shared helpers and the account check |

```bash
infra/ecr.sh preempt --context . --dockerfile products/preempt/Dockerfile
infra/apprunner.sh preempt                 # prints https://<id>.awsapprunner.com
infra/apprunner.sh preempt --status
```

App Runner is x86 only, so `ecr.sh` builds `linux/amd64` by default.
`apprunner.sh --delete` asks before deleting.
