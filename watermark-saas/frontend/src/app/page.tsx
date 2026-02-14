"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { requestUpload, uploadFile, getJobStatus, requestDownload } from "@/lib/api";

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/webp"];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
const POLL_INTERVAL = 2000;
const CREDITS_KEY = "watermark_credits_remaining";

type AppState = "idle" | "uploading" | "processing" | "completed" | "failed";

export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [credits, setCredits] = useState(3);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  // Load credits from localStorage
  useEffect(() => {
    const stored = localStorage.getItem(CREDITS_KEY);
    if (stored !== null) {
      setCredits(parseInt(stored, 10));
    }
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const resetState = useCallback(() => {
    setState("idle");
    setFile(null);
    setJobId(null);
    setPreviewUrl(null);
    setError(null);
    setUploadProgress(0);
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  const validateFile = (f: File): string | null => {
    if (!ACCEPTED_TYPES.includes(f.type)) {
      return "Unsupported file type. Please upload PNG, JPG, or WEBP.";
    }
    if (f.size > MAX_FILE_SIZE) {
      return "File too large. Maximum size is 50MB.";
    }
    return null;
  };

  const handleFile = async (selectedFile: File) => {
    const validationError = validateFile(selectedFile);
    if (validationError) {
      setError(validationError);
      return;
    }

    setFile(selectedFile);
    setError(null);
    setState("uploading");
    setUploadProgress(10);

    try {
      // Request presigned URL
      const { jobId: newJobId, uploadUrl, remainingCredits } =
        await requestUpload(selectedFile.name, selectedFile.type);

      setJobId(newJobId);
      setUploadProgress(30);

      // Upload file to S3
      await uploadFile(uploadUrl, selectedFile);
      setUploadProgress(100);

      // Update credits
      setCredits(remainingCredits);
      localStorage.setItem(CREDITS_KEY, String(remainingCredits));

      // Start polling for status
      setState("processing");
      startPolling(newJobId);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Upload failed";
      setError(message);
      setState("failed");
    }
  };

  const startPolling = (id: string) => {
    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      try {
        const status = await getJobStatus(id);

        if (status.status === "completed") {
          if (pollRef.current) clearInterval(pollRef.current);
          setPreviewUrl(status.previewUrl || null);
          setState("completed");
        } else if (status.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
          setError(status.error || "Processing failed");
          setState("failed");
        }
      } catch {
        // Polling errors are transient; keep retrying
      }
    }, POLL_INTERVAL);
  };

  const handleDownload = async () => {
    if (!jobId) return;
    try {
      const { downloadUrl } = await requestDownload(jobId);
      window.open(downloadUrl, "_blank");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Download failed";
      setError(message);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) handleFile(droppedFile);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  return (
    <main className="min-h-screen flex flex-col items-center px-4 py-12">
      {/* Hero */}
      <div className="max-w-2xl text-center mb-10">
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          NotebookLM Watermark Remover
        </h1>
        <p className="mt-4 text-lg text-gray-600">
          Remove watermarks from your NotebookLM screenshots instantly.
          Upload an image and get a clean result in seconds.
        </p>
        <div className="mt-4 flex justify-center gap-6 text-sm text-gray-500">
          <span>PNG / JPG / WEBP</span>
          <span>Up to 50MB</span>
          <span>Processed in seconds</span>
        </div>
      </div>

      {/* Credits indicator */}
      <div className="mb-6 text-sm text-gray-500">
        {credits > 0
          ? `${credits} of 3 free uses remaining today`
          : "Daily free limit reached. Try again tomorrow."}
      </div>

      {/* Upload / Status area */}
      <div className="w-full max-w-xl">
        {state === "idle" && (
          <div
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer ${
              dragActive
                ? "border-blue-500 bg-blue-50"
                : "border-gray-300 hover:border-gray-400"
            } ${credits <= 0 ? "opacity-50 pointer-events-none" : ""}`}
            onDrop={handleDrop}
            onDragOver={handleDrag}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.webp"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M12 16v-8m0 0l-3 3m3-3l3 3M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1"
              />
            </svg>
            <p className="mt-4 text-gray-600">
              <span className="font-medium text-blue-600">Click to upload</span>{" "}
              or drag and drop
            </p>
            <p className="mt-1 text-sm text-gray-400">
              PNG, JPG, or WEBP up to 50MB
            </p>
          </div>
        )}

        {state === "uploading" && (
          <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
            <p className="text-gray-700 font-medium mb-4">
              Uploading {file?.name}...
            </p>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
            <p className="mt-2 text-sm text-gray-500">{uploadProgress}%</p>
          </div>
        )}

        {state === "processing" && (
          <div className="rounded-xl border border-gray-200 bg-white p-8 text-center">
            <div className="inline-block animate-spin rounded-full h-10 w-10 border-4 border-gray-200 border-t-blue-600 mb-4" />
            <p className="text-gray-700 font-medium">
              Processing your image...
            </p>
            <p className="mt-1 text-sm text-gray-500">
              This usually takes a few seconds.
            </p>
          </div>
        )}

        {state === "completed" && (
          <div className="rounded-xl border border-gray-200 bg-white p-6 text-center">
            <p className="text-green-600 font-medium text-lg mb-4">
              Watermark removed successfully
            </p>
            {previewUrl && (
              <div className="mb-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt="Preview of cleaned image"
                  className="max-w-full rounded-lg border border-gray-100 mx-auto"
                />
                <p className="mt-1 text-xs text-gray-400">Low-res preview</p>
              </div>
            )}
            <div className="flex gap-3 justify-center">
              <button
                onClick={handleDownload}
                className="px-6 py-2.5 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
              >
                Download Full Resolution
              </button>
              <button
                onClick={resetState}
                className="px-6 py-2.5 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
              >
                Process Another
              </button>
            </div>
          </div>
        )}

        {state === "failed" && (
          <div className="rounded-xl border border-red-200 bg-red-50 p-8 text-center">
            <p className="text-red-600 font-medium mb-2">
              Something went wrong
            </p>
            <p className="text-sm text-red-500 mb-4">{error}</p>
            <button
              onClick={resetState}
              className="px-6 py-2.5 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-white transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Error toast (for non-fatal errors) */}
        {error && state === "idle" && (
          <p className="mt-3 text-sm text-red-500 text-center">{error}</p>
        )}
      </div>

      {/* Footer */}
      <footer className="mt-auto pt-12 pb-6 text-center text-xs text-gray-400">
        Files are automatically deleted after 24 hours. We do not store or share
        your data.
      </footer>
    </main>
  );
}
