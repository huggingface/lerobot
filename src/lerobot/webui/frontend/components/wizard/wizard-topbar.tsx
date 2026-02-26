"use client";

import { RotateCcw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useWizard } from "./wizard-provider";

export function WizardTopbar() {
  const { clearAllValues, restart } = useWizard();

  return (
    <div className="sticky top-0 z-10 flex h-14 items-center justify-end border-b bg-white px-6">
      <div className="flex gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={clearAllValues}
          className="text-muted-foreground"
        >
          <Trash2 className="mr-1.5 h-4 w-4" />
          Clear Values
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={restart}
        >
          <RotateCcw className="mr-1.5 h-4 w-4" />
          Restart
        </Button>
      </div>
    </div>
  );
}
