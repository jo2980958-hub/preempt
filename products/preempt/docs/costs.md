# What was created on AWS, and what it costs

**Live endpoint: <https://2uhvgzwrwc.us-east-2.awsapprunner.com>**

Everything is tagged `Project=opencv26`. Account <aws-account-id>.

## What exists

| Resource | Region | Name | Configuration |
|---|---|---|---|
| App Runner service | us-east-2 | `opencv26-preempt` | 2 vCPU, 4 GB, port 8000, always on |
| ECR repository | us-east-2 | `opencv26/preempt` | 180 MB compressed (651 MB on disk), lifecycle keeps 10 images and expires untagged after a day |
| ECR repository | us-east-1 | `opencv26/preempt` | **one build behind** (digest `13d2594`, no `PREEMPT_WEB_DIR`). Left from the first deploy. Do not deploy from it without re-pushing: it serves the shared shell, not Preempt |
| IAM role | global | `opencv26-apprunner-ecr-access` | lets App Runner pull from ECR, nothing else |
| CloudWatch log groups | us-east-2 | `/aws/apprunner/opencv26-preempt/*` | application and service logs |

No load balancer, no S3 bucket in the request path, no database, no NAT gateway,
no VPC connector.

## What it costs

App Runner charges a provisioned rate for held memory and an active rate for vCPU
while a request is being served. At the published us-east-2 rates:

| Line | Rate | Hourly | Monthly |
|---|---|---|---|
| Provisioned memory, 4 GB, always on | $0.007 per GB-hour | $0.0280 | **$20.44** |
| Active vCPU, 2 vCPU, while serving | $0.064 per vCPU-hour | $0.128 while active | a few cents at demo traffic |
| ECR storage, about 0.36 GB across two regions | $0.10 per GB-month | — | $0.04 |
| CloudWatch logs | $0.50 per GB ingested | — | under $0.10 |
| **Total at rest** | | **about $0.028** | **about $20.60** |

A judge running every bundled sample and uploading a video of their own adds a few
minutes of active vCPU, which is under ten cents.

The service is **left running**. Scaling it to zero would save about $20 a month
and would cost a judge a cold start, which is the wrong trade for a two-week
judging window.

## Two decisions worth recording

**us-east-2 rather than us-east-1, and Preempt is the only one of the five.**
The brief said us-east-1, and the four sibling entries are there. This account is
restricted to two App Runner services per region, and both us-east-1 slots were
already held by other services when this was deployed. The error, verbatim:

> `Account <aws-account-id> is restricted and can support only two App Runner
> services per region at the moment.`

us-east-2 was the nearest region with a free slot. So an architecture diagram for
the project as a whole has four products in us-east-1 and this one in us-east-2;
the diagram in `architecture.md` shows us-east-2 because that is where this
service actually runs.

The image was originally pushed to both regions, because App Runner can only pull
a private image from its own region. Only the us-east-2 copy has been kept current
since; see the table above.

**2 vCPU and 4 GB rather than the cheapest option.** The image carries two ONNX
models and runs both per frame on an uploaded video. At 0.25 vCPU a judge's video
would take several minutes; at 2 vCPU the pipeline runs faster than real time, and
a bundled pose track returns in well under a second.

## Cold start

Measured locally on the same image: the container answers `/healthz` **2 seconds**
after start, and a bundled sample analysed end to end returns in under a second.
Nothing is fetched at runtime. Both ONNX models, all seven sample pose tracks and
the default room are baked into the image, so the endpoint works with no network
beyond the request itself.

## Turning it off

```bash
AWS_REGION=us-east-2 infra/apprunner.sh preempt --delete
aws ecr delete-repository --region us-east-2 --repository-name opencv26/preempt --force
aws ecr delete-repository --region us-east-1 --repository-name opencv26/preempt --force
```
