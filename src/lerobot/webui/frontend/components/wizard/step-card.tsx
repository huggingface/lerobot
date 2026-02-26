"use client";

import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useWizard } from "./wizard-provider";

interface StepCardProps {
  title: string;
  description: string;
  children: React.ReactNode;
  showNext?: boolean;
  nextDisabled?: boolean;
  nextLabel?: string;
}

export function StepCard({
  title,
  description,
  children,
  showNext = true,
  nextDisabled = false,
  nextLabel = "Continue",
}: StepCardProps) {
  const { goNext } = useWizard();

  return (
    <Card className="border shadow-sm">
      <CardHeader>
        <CardTitle className="text-xl">{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent>{children}</CardContent>
      {showNext && (
        <CardFooter className="flex justify-end border-t pt-6">
          <Button onClick={goNext} disabled={nextDisabled}>
            {nextLabel}
          </Button>
        </CardFooter>
      )}
    </Card>
  );
}
