#!/usr/bin/env bash
set -e -o errtrace
trap onError ERR
onError(){ echo "$(tput setaf 1)Error code [$?] occurred at (latest on top)$(tput sgr0):"; while caller $((n++)); do :; done; }

readonly DOCKER_IMAGE=registry.corp.ailabs.tw/smart-city/cpl-platform/research/sfm-lab:`git rev-parse --abbrev-ref HEAD`
readonly K8S_FILE=deploy/sfm.yaml
readonly BRANCH=`git rev-parse --abbrev-ref HEAD | sed "s/_/-/g" `

case ${1} in

k8srun)
  if [[ -z ${2} ]]; then
    echo "Usage: ${0} ${1} {k8s-hostname}"
    exit 1
  fi
  sed "s/<---USER--->/${USER}/g; s/<---HOSTNAME--->/${2}/g; s/<---BRANCH--->/${BRANCH}/g" $K8S_FILE | kubectl apply -f -
  ;;

k8sbash)
  if [[ -z ${2} ]]; then
    echo "Usage: ${0} ${1} {k8s-podname}"
    exit 1
  fi
  kubectl exec -it ${2} -- /bin/bash
  ;;

k8sls)
  kubectl get pod | grep "^sfm-"
  ;;

k8sdel)
  if [[ -z ${2} ]]; then
    echo "Usage: ${0} ${1} {k8s-hostname}"
    exit 1
  fi
  sed "s/<---USER--->/${USER}/g; s/<---HOSTNAME--->/${2}/g" $K8S_FILE | kubectl delete -f -
  ;;

build)
  DOCKER_BUILDKIT=1 docker build -t ${DOCKER_IMAGE} .
  ;;

push)
  docker push ${DOCKER_IMAGE}
  ;;

run)
  docker run -it --rm -v `pwd`/src:/root/src ${DOCKER_IMAGE}
  ;;

*)
  echo "Usage: ${0} k8srun|k8sdel|k8sls|k8sbash|build|push|run"
  ;;

esac
