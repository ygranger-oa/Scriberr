import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from "@/components/ui/dialog";
import { FileText } from "lucide-react";
import { useLogs } from "@/features/transcription/hooks/useAudioDetail";

interface LogsDialogProps {
    audioId: string;
    isOpen: boolean;
    onClose: (open: boolean) => void;
}

export function LogsDialog({ audioId, isOpen, onClose }: LogsDialogProps) {
    const { data: logsContent, isLoading } = useLogs(audioId);

    return (
        <Dialog open={isOpen} onOpenChange={onClose}>
            <DialogContent className="grid h-[calc(100dvh-1rem)] w-[calc(100vw-1rem)] max-w-6xl grid-rows-[auto_minmax(0,1fr)] gap-0 overflow-hidden border-[var(--border-subtle)] bg-[var(--bg-card)] p-4 shadow-[var(--shadow-float)] sm:h-[min(90dvh,56rem)] sm:max-w-6xl sm:p-6">
                <DialogHeader className="border-b border-[var(--border-subtle)] pb-4 pr-8">
                    <DialogTitle className="text-[var(--text-primary)] flex items-center gap-2 text-xl font-bold tracking-tight">
                        <FileText className="h-5 w-5 text-[var(--brand-solid)]" />
                        Transcription Logs
                    </DialogTitle>
                    <DialogDescription className="text-[var(--text-secondary)]">
                        System output and processing events.
                    </DialogDescription>
                </DialogHeader>

                <div className="min-h-0 min-w-0 overflow-hidden pt-4">
                    {isLoading ? (
                        <div className="flex h-full flex-col items-center justify-center gap-4 py-12">
                            <div className="h-8 w-8 border-4 border-[var(--brand-solid)] border-t-transparent rounded-full animate-spin" />
                            <span className="text-[var(--text-tertiary)] animate-pulse">Loading logs...</span>
                        </div>
                    ) : logsContent?.available === false ? (
                        <div className="flex h-full items-center justify-center py-12 text-center text-[var(--text-tertiary)]">
                            No logs available for this transcription job.
                        </div>
                    ) : (
                        <pre className="h-full w-full max-w-full overflow-auto whitespace-pre-wrap break-words rounded-[var(--radius-card)] border border-white/10 bg-[#0A0A0A] p-4 font-mono text-xs leading-relaxed text-[#EDEDED] shadow-inner [overflow-wrap:anywhere] sm:text-sm">
                            {logsContent?.content || "No logs available."}
                        </pre>
                    )}
                </div>
            </DialogContent>
        </Dialog>
    );
}
