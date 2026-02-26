"use client";

import { Check } from "lucide-react";
import { cn } from "@/lib/utils";
import { STEPS } from "@/lib/wizard-types";
import { useWizard } from "./wizard-provider";

export function WizardSidebar() {
  const { state, goToStep } = useWizard();

  return (
    <aside className="fixed left-0 top-0 z-20 flex h-screen w-60 flex-col border-r bg-white">
      <div className="flex h-14 items-center border-b px-6">
        <span className="text-lg font-semibold tracking-tight">xLeRobot</span>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {STEPS.map((step, i) => {
          const isCurrent = i === state.currentStep;
          const isComplete = state.completedSteps[i];

          return (
            <button
              key={i}
              onClick={() => goToStep(i)}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors",
                isCurrent && "bg-primary/5 font-medium text-foreground",
                !isCurrent &&
                  "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <StepIndicator
                step={i}
                isCurrent={isCurrent}
                isComplete={isComplete}
              />
              <span>{step.label}</span>
            </button>
          );
        })}
      </nav>
      <div className="border-t px-6 py-4">
        <p className="text-xs text-muted-foreground">LeRobot Setup Wizard</p>
      </div>
    </aside>
  );
}

function StepIndicator({
  step,
  isCurrent,
  isComplete,
}: {
  step: number;
  isCurrent: boolean;
  isComplete: boolean;
}) {
  if (isComplete && !isCurrent) {
    return (
      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <Check className="h-3.5 w-3.5" />
      </div>
    );
  }

  if (isCurrent) {
    return (
      <div className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-primary">
        <div className="h-2 w-2 rounded-full bg-primary" />
      </div>
    );
  }

  return (
    <div className="flex h-6 w-6 items-center justify-center rounded-full border-2 border-muted-foreground/30">
      <span className="text-xs text-muted-foreground">{step + 1}</span>
    </div>
  );
}
