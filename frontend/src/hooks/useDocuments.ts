import { useCallback, useEffect, useState } from "react";
import * as api from "../lib/api";
import type { DocumentInfo } from "../lib/types";

const SAMPLE = "/samples/northwind_handbook.pdf";

/** The document list, kept fresh by polling: fast while something is being ingested. */
export function useDocuments(onError: (message: string | null) => void) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);

  const refresh = useCallback(() => api.listDocuments().then(setDocuments).catch(() => {}), []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const working = documents.some((d) => d.status === "queued" || d.status === "processing");
  useEffect(() => {
    // Slow polling still picks up uploads from other tabs or the API.
    const timer = setInterval(refresh, working ? 700 : 5000);
    return () => clearInterval(timer);
  }, [working, refresh]);

  const upload = useCallback(
    async (files: File[], options: api.UploadOptions) => {
      onError(null);
      for (const file of files) {
        try {
          await api.uploadDocument(file, file.name, options);
        } catch (error) {
          onError(`${file.name}: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
      await refresh();
    },
    [onError, refresh],
  );

  const uploadSample = useCallback(async () => {
    const blob = await fetch(SAMPLE).then((r) => r.blob());
    await upload([new File([blob], "northwind_handbook.pdf", { type: "application/pdf" })], {});
  }, [upload]);

  const remove = useCallback(
    async (id: string) => {
      await api.deleteDocument(id);
      await refresh();
    },
    [refresh],
  );

  return { documents, upload, uploadSample, remove };
}
