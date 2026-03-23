const API = "/api";

export type UploadResponse = {
  job_id: string;
  folder_id: string;
  task: string;
};

export type JobStatus = {
  job_id: string;
  status: string;
  result: string | null;
  error: string | null;
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

export async function getModelDownloadUrl(folderId: string): Promise<string> {
  const res = await fetch(`${API}/models/${folderId}/download`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.download_url;
}
