{{/*
Shared pieces. Every pod gets the same security settings from here, so a new component can't
forget one: tests/test_k8s_chart.py renders the chart and checks every pod.
*/}}

{{- define "wikiguard.labels" -}}
app.kubernetes.io/part-of: wikiguard
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{/* The component's name. wikiguard.labels adds the rest, so the two never repeat a key. */}}
{{- define "wikiguard.selector" -}}
app.kubernetes.io/name: {{ . }}
{{- end }}

{{/* An image by digest when CD has recorded one (images.yaml), otherwise by tag. */}}
{{- define "wikiguard.image" -}}
{{- $root := index . 0 -}}
{{- $name := index . 1 -}}
{{- $repo := printf "%s/%s-%s" $root.Values.image.registry $root.Values.image.prefix $name -}}
{{- $digest := index ($root.Values.digests | default dict) $name | default "" -}}
{{- if $digest -}}
{{ $repo }}@{{ $digest }}
{{- else -}}
{{ $repo }}:{{ $root.Values.image.tag }}
{{- end -}}
{{- end }}

{{/* Pod Security "restricted", plus a fixed non-root user that matches every image's USER. */}}
{{- define "wikiguard.podSecurity" -}}
runAsNonRoot: true
runAsUser: {{ .user | default 10001 }}
runAsGroup: {{ .user | default 10001 }}
fsGroup: {{ .user | default 10001 }}
fsGroupChangePolicy: OnRootMismatch
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{- define "wikiguard.containerSecurity" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
privileged: false
capabilities:
  drop: [ALL]
{{- end }}

{{/* What every container reads: the shared settings, then its own Secret. */}}
{{- define "wikiguard.envFrom" -}}
- configMapRef:
    name: wikiguard-config
- secretRef:
    name: wikiguard-{{ . }}
{{- end }}

{{- define "wikiguard.env" -}}
{{- range $key, $value := . }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}

{{/* Named volumes a component can ask for, mounted where the code expects them. */}}
{{- define "wikiguard.volumeMounts" -}}
- name: tmp
  mountPath: /tmp
{{- range . }}
{{- if eq . "archive" }}
- name: archive
  mountPath: /archive
{{- else if eq . "archiveReadOnly" }}
- name: archive
  mountPath: /archive
  readOnly: true
{{- else if eq . "ingestorState" }}
- name: ingestor-state
  mountPath: /state
{{- else if eq . "models" }}
- name: models
  mountPath: /models/cache
{{- else if eq . "backups" }}
- name: backups
  mountPath: /backups
{{- end }}
{{- end }}
{{- end }}

{{- define "wikiguard.volumes" -}}
- name: tmp
  emptyDir:
    sizeLimit: 1Gi
{{- range . }}
{{- if or (eq . "archive") (eq . "archiveReadOnly") }}
- name: archive
  persistentVolumeClaim:
    claimName: wikiguard-archive
{{- else if eq . "ingestorState" }}
- name: ingestor-state
  persistentVolumeClaim:
    claimName: wikiguard-ingestor-state
{{- else if eq . "models" }}
- name: models
  persistentVolumeClaim:
    claimName: wikiguard-models
{{- else if eq . "backups" }}
- name: backups
  persistentVolumeClaim:
    claimName: wikiguard-backups
{{- end }}
{{- end }}
{{- end }}

{{/*
A shallow clone of the public repo, then the golden set from DVC: what the trainer and the
snapshot purge work in. Runs as an init container from the same image, which has git and DVC.
DVC reads DagsHub's key from AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the component's Secret.
*/}}
{{- define "wikiguard.checkout" -}}
- name: checkout
  image: {{ .image }}
  imagePullPolicy: {{ .root.Values.image.pullPolicy }}
  securityContext:
    {{- include "wikiguard.containerSecurity" . | nindent 4 }}
  envFrom:
    {{- include "wikiguard.envFrom" .secret | nindent 4 }}
  env:
    - name: HOME
      value: /tmp
    - name: REPO_URL
      value: {{ .root.Values.repo.url | quote }}
    - name: REPO_BRANCH
      value: {{ .root.Values.repo.branch | quote }}
    - name: CHECKOUT
      value: {{ .path | quote }}
  command:
    - sh
    - -ec
    - |
      if [ -d "$CHECKOUT/.git" ]; then
        git -C "$CHECKOUT" fetch --depth 1 origin "$REPO_BRANCH"
        git -C "$CHECKOUT" reset --hard "origin/$REPO_BRANCH"
      else
        git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$CHECKOUT"
      fi
      cd "$CHECKOUT"
      dvc config --local core.analytics false
      dvc pull {{ .pull }}
  volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: work
      mountPath: {{ .mount }}
{{- end }}
