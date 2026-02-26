"use client";

import { Check, Hand, User } from "lucide-react";
import { cn } from "@/lib/utils";
import type { RobotMode } from "@/lib/wizard-types";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

export function RobotTypeStep() {
  const { state, dispatch } = useWizard();

  function selectMode(mode: RobotMode) {
    dispatch({ type: "SET_ROBOT_MODE", mode });
  }

  return (
    <StepCard
      title="Choose Robot Type"
      description="Select your robot arm configuration."
      nextDisabled={!state.robotMode}
    >
      <div className="grid grid-cols-2 gap-4">
        <SelectableCard
          selected={state.robotMode === "single"}
          onClick={() => selectMode("single")}
          icon={<User className="h-8 w-8" />}
          title="Single Arm"
          description="One follower + one leader arm"
        />
        <SelectableCard
          selected={state.robotMode === "bimanual"}
          onClick={() => selectMode("bimanual")}
          icon={<Hand className="h-8 w-8" />}
          title="Bimanual"
          description="Two follower + two leader arms"
        />
      </div>
    </StepCard>
  );
}

function SelectableCard({
  selected,
  onClick,
  icon,
  title,
  description,
}: {
  selected: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "relative flex flex-col items-center gap-3 rounded-xl border-2 p-6 text-center transition-all",
        selected
          ? "border-primary bg-primary/5 ring-2 ring-primary/20"
          : "border-border hover:border-primary/40 hover:bg-muted/50"
      )}
    >
      {selected && (
        <div className="absolute right-3 top-3 flex h-5 w-5 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Check className="h-3 w-3" />
        </div>
      )}
      <div className="text-muted-foreground">{icon}</div>
      <div>
        <p className="font-medium">{title}</p>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
    </button>
  );
}
