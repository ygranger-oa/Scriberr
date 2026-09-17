const commitDate = import.meta.env.VITE_COMMIT_DATE

function formatCommitDate(value: string | undefined): string | null {
    if (!value) return null

    const date = new Date(value)
    if (Number.isNaN(date.getTime())) return null

    return new Intl.DateTimeFormat(undefined, {
        day: "numeric",
        month: "short",
        year: "numeric",
    }).format(date)
}

export function StatusBar() {
    const formattedDate = formatCommitDate(commitDate)

    return (
        <footer className="fixed inset-x-0 bottom-0 z-40 border-t border-[var(--border)] bg-[var(--bg-main)]/95 px-4 py-1.5 text-center text-xs text-[var(--text-secondary)] backdrop-blur-sm">
            {formattedDate ? `Version du ${formattedDate}` : "Version de développement"}
        </footer>
    )
}
