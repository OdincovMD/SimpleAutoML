const API = "/api";

export type UploadResponse = {
  job_id: string;
  folder_id: string;
  task: string;
};

export type JobProgress = {
  kind?: "train" | "inference";
  inference_id?: string;
  step?: string;
  steps_history?: string[];
  folder_id?: string;
  task_type?: string;
};

export type JobTrainResult = {
  kind: "train";
  folder_id: string;
  task_type: string;
  step: "job_complete";
  steps_history: string[];
};

export type JobInferenceResult = {
  kind: "inference";
  folder_id: string;
  task_type: string;
  inference_id: string;
  step: "infer_complete";
  steps_history: string[];
};

export function isJobTrainComplete(
  p: JobProgress | null,
  status: string
): p is JobTrainResult {
  return (
    status === "SUCCESS" &&
    p != null &&
    p.kind === "train" &&
    p.step === "job_complete" &&
    typeof p.folder_id === "string" &&
    typeof p.task_type === "string" &&
    Array.isArray(p.steps_history)
  );
}

export function isJobInferenceComplete(
  p: JobProgress | null,
  status: string
): p is JobInferenceResult {
  return (
    status === "SUCCESS" &&
    p != null &&
    p.kind === "inference" &&
    p.step === "infer_complete" &&
    typeof p.folder_id === "string" &&
    typeof p.inference_id === "string" &&
    typeof p.task_type === "string" &&
    Array.isArray(p.steps_history)
  );
}

export type JobStatus = {
  job_id: string;
  status: string;
  progress: JobProgress | null;
  result: null;
  error: string | null;
  completed_at?: string | null;
};

export type DriveFolder = { id: string; name: string };

export async function listDriveFolders(
  parentName?: string,
  parentId?: string
): Promise<DriveFolder[]> {
  const params = new URLSearchParams();
  if (parentName?.trim()) params.set("parent_name", parentName.trim());
  if (parentId) params.set("parent_id", parentId);
  const qs = params.toString();
  const res = await fetch(`${API}/drive/folders${qs ? `?${qs}` : ""}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API}/datasets/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function startJobFromDrive(
  folderId: string
): Promise<UploadResponse> {
  const res = await fetch(`${API}/datasets/from-drive`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder_id: folderId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${API}/jobs/${jobId}/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export type ModelListItem = {
  train_folder: string;
  version: number;
  imgsz: number;
  task_type: string | null;
  classes: string[];
  trained_at?: string | null;
};

export type DatasetMeta = {
  folder_id: string;
  files_total: number;
  files_pending_train: number;
  has_model: boolean;
  task_type: string | null;
};

export async function listModels(): Promise<ModelListItem[]> {
  const res = await fetch(`${API}/models`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDatasetMeta(folderId: string): Promise<DatasetMeta> {
  const res = await fetch(
    `${API}/datasets/${encodeURIComponent(folderId)}/meta`
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function retrainDataset(
  folderId: string,
  file: File,
  taskType?: string
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  if (taskType?.trim()) form.append("task_type", taskType.trim());
  const res = await fetch(
    `${API}/datasets/${encodeURIComponent(folderId)}/retrain`,
    { method: "POST", body: form }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function startInference(
  folderId: string,
  taskType: string,
  file: File
): Promise<{
  job_id: string;
  folder_id: string;
  task_type: string;
  inference_id: string;
  inference_upload_id: string;
}> {
  const form = new FormData();
  form.append("file", file);
  form.append("task_type", taskType);
  const res = await fetch(
    `${API}/inference/${encodeURIComponent(folderId)}`,
    { method: "POST", body: form }
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getInferenceDownloadUrl(
  folderId: string,
  inferenceId: string
): Promise<string> {
  const res = await fetch(
    `${API}/inference/${encodeURIComponent(folderId)}/download/${encodeURIComponent(inferenceId)}`
  );
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.download_url as string;
}

export async function getModelDownloadUrl(folderId: string): Promise<string> {
  const res = await fetch(
    `${API}/models/${encodeURIComponent(folderId)}/download`
  );
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.download_url as string;
}
